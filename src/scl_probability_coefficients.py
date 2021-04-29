import argparse
import ee
import os
import re
import subprocess
import numpy as np
import pandas as pd
import pyodbc
import time
import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional, Union
from geomet import wkt
from google.cloud.storage import Client
from google.cloud.exceptions import NotFound
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from task_base import SCLTask


class ConversionException(Exception):
    pass


class SCLProbabilityCoefficients(SCLTask):
    GRID_LABEL = "GridName"
    CELL_LABEL = "GridCellCode"
    MASTER_GRID_LABEL = "mastergrid"
    MASTER_CELL_LABEL = "mastergridcell"
    MASTER_CELL_ID_LABEL = "id"
    POINT_LOC_LABEL = "PointLocation"
    GRIDCELL_LOC_LABEL = "GridCellLocation"
    ZONES_LABEL = "Biome_zone"
    UNIQUE_ID_LABEL = "UniqueID"
    EE_NODATA = -9999
    BUCKET = "scl-pipeline"

    MASTERGRID_DF_COLUMNS = [UNIQUE_ID_LABEL, MASTER_GRID_LABEL, MASTER_CELL_LABEL]

    google_creds_path = "/.google_creds"
    use_cache = True
    save_cache = True  # only relevant when use_cache = False
    inputs = {
        "obs_adhoc": {"maxage": 50},
        "obs_ss": {"maxage": 50},
        "obs_ct": {"maxage": 50},
        "hii": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "projects/HII/v1/hii",
            "maxage": 30,
        },
        "tri": {
            "ee_type": SCLTask.IMAGE,
            "ee_path": "projects/SCL/v1/tri_aster",
            "static": True,
        },
        # original dynamic calculation of tri from this dem using method below
        # "dem": {"ee_type": SCLTask.IMAGE, "ee_path": "CGIAR/SRTM90_V4", "static": True},
        # TODO: replace with roads from OSM and calculate distance (Kim 1)
        "roads": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/Panthera-Earth-Engine/Roads/SouthAsiaRoads",
            "maxage": 1,
        },
        "structural_habitat": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/canonical/structural_habitat",
            "maxage": 10,  # until we have full-range SH for every year
        },
        "zones": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/zones",
            "static": True,
        },
        "gridcells": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/covar_gridcells",
            "static": True,
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.OBSDB_HOST = os.environ["OBSDB_HOST"]
            self.OBSDB_NAME = os.environ["OBSDB_NAME"]
            self.OBSDB_USER = os.environ["OBSDB_USER"]
            self.OBSDB_PASS = os.environ["OBSDB_PASS"]
        except KeyError as e:
            self.status = self.FAILED
            raise KeyError(str(e)) from e

        _obsconn_str = (
            f"DRIVER=FreeTDS;SERVER={self.OBSDB_HOST};PORT=1433;DATABASE="
            f"{self.OBSDB_NAME};UID={self.OBSDB_USER};PWD={self.OBSDB_PASS}"
        )
        self.obsconn = pyodbc.connect(_obsconn_str)

        # Set up google cloud credentials separate from ee creds
        creds_path = Path(self.google_creds_path)
        if creds_path.exists() is False:
            with open(str(creds_path), "w") as f:
                f.write(self.service_account_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_creds_path

        self._zone_ids = []
        self._grids = {}
        self._df_adhoc = None
        self._df_ct = None
        self._df_ss = None
        self._df_covars = None
        self.zone = None
        self.Nx = 0
        self.Nw = 0
        # TODO: set these dynamically; right now assumes constant detection probability for SS/CT data
        self.Npsign = 1
        self.NpCT = 1
        # coefficients relevant to presence-only and background detection only
        self.po_detection_covars = None
        # coefficients relevant to occupancy, shared across models
        self.presence_covars = None
        self.psi = []
        self.df_zeta = pd.DataFrame(
            columns = ["zeta0","zeta1"],
            index = None
        )

        self.zones = ee.FeatureCollection(self.inputs["zones"]["ee_path"])
        self.gridcells = ee.FeatureCollection(self.inputs["gridcells"]["ee_path"])
        self.fc_csvs = []

    def _download_from_cloudstorage(self, blob_path: str, local_path: str) -> str:
        client = Client()
        bucket = client.get_bucket(self.BUCKET)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return local_path

    def _upload_to_cloudstorage(self, local_path: str, blob_path: str) -> str:
        client = Client()
        bucket = client.bucket(self.BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, timeout=3600)
        return blob_path

    def _remove_from_cloudstorage(self, blob_path: str):
        client = Client()
        bucket = client.bucket(self.BUCKET)
        try:  # don't fail entire task if this fails
            bucket.delete_blob(blob_path)
        except NotFound:
            print(f"{blob_path} not found")

    def _parse_task_id(self, output: Union[str, bytes]) -> Optional[str]:
        if isinstance(output, bytes):
            text = output.decode("utf-8")
        else:
            text = output

        task_id_regex = re.compile(r"(?<=ID: ).*", flags=re.IGNORECASE)
        try:
            matches = task_id_regex.search(text)
            if matches is None:
                return None
            return matches[0]
        except TypeError:
            return None

    def _cp_storage_to_ee_table(
        self, blob_uri: str, table_asset_id: str, geofield: str = "geom"
    ) -> str:
        try:
            cmd = [
                "/usr/local/bin/earthengine",
                f"--service_account_file={self.google_creds_path}",
                "upload table",
                f"--primary_geometry_column {geofield}",
                f"--asset_id={table_asset_id}",
                blob_uri,
            ]
            output = subprocess.check_output(
                " ".join(cmd), stderr=subprocess.STDOUT, shell=True
            )
            task_id = self._parse_task_id(output)
            if task_id is None:
                raise TypeError("task_id is None")
            self.ee_tasks[task_id] = {}
            return task_id
        except subprocess.CalledProcessError as err:
            raise ConversionException(err.stdout)

    def _get_df(self, query):
        _scenario_clause = (
            f"AND ScenarioName IS NULL OR ScenarioName = '{self.CANONICAL}'"
        )
        if self.scenario and self.scenario != self.CANONICAL:
            _scenario_clause = f"AND ScenarioName = '{self.scenario}'"

        query = f"{query} {_scenario_clause}"
        df = pd.read_sql(query, self.obsconn)
        return df

    def _obs_feature(self, point_geom, gridcell_geom, id_label):
        geom = gridcell_geom
        if point_geom:
            geom = point_geom
        return ee.Feature(wkt.loads(geom), {self.UNIQUE_ID_LABEL: id_label})

    def _find_master_grid_cell(self, obs_feature):
        centroid = obs_feature.centroid().geometry()
        intersects = ee.Filter.intersects(".geo", None, ".geo")
        matching_zones = ee.Join.simple().apply(self.zones, obs_feature, intersects)
        zone_id_true = ee.Number(matching_zones.first().get(self.ZONES_LABEL))
        id_false = ee.Number(self.EE_NODATA)
        zone_id = ee.Number(
            ee.Algorithms.If(matching_zones.size().gte(1), zone_id_true, id_false)
        )

        matching_gridcells = self.gridcells.filter(
            ee.Filter.eq("zone", zone_id)
        ).filterBounds(centroid)
        gridcell_id_true = ee.Number(
            matching_gridcells.first().get(self.MASTER_CELL_ID_LABEL)
        )
        gridcell_id = ee.Number(
            ee.Algorithms.If(
                zone_id.neq(self.EE_NODATA),
                ee.Algorithms.If(
                    matching_gridcells.size().gte(1), gridcell_id_true, id_false
                ),
                id_false,
            )
        )

        obs_feature = obs_feature.setMulti(
            {self.MASTER_GRID_LABEL: zone_id, self.MASTER_CELL_LABEL: gridcell_id}
        )

        return obs_feature

    # add "mastergrid" and "mastergridcell" to df
    def zonify(self, df):
        obs_features = ee.FeatureCollection(
            [
                self._obs_feature(o[0], o[1], o[2])
                for o in zip(
                    df[self.POINT_LOC_LABEL],
                    df[self.GRIDCELL_LOC_LABEL],
                    df[self.UNIQUE_ID_LABEL],
                )
                if (o[0] or o[1]) and o[2]
            ]
        )

        return_obs_features = obs_features.map(self._find_master_grid_cell)
        master_grid_df = self.fc2df(return_obs_features, self.MASTERGRID_DF_COLUMNS)
        if master_grid_df.empty:
            master_grid_df[self.UNIQUE_ID_LABEL] = pd.Series(dtype="object")
            master_grid_df[self.MASTER_GRID_LABEL] = pd.Series(dtype="object")
            master_grid_df[self.MASTER_CELL_LABEL] = pd.Series(dtype="object")

        df = pd.merge(left=df, right=master_grid_df)

        # save out non-intersecting observations
        df_nonintersections = df[
            (df[self.MASTER_GRID_LABEL] == self.EE_NODATA)
            | (df[self.MASTER_CELL_LABEL] == self.EE_NODATA)
        ]
        if not df_nonintersections.empty:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            df_nonintersections.to_csv(f"nonintersecting-{timestr}.csv")

        # Filter out rows not in any zone and rows not in any gridcell (-9999)
        df = df[
            (df[self.MASTER_GRID_LABEL] != self.EE_NODATA)
            & (df[self.MASTER_CELL_LABEL] != self.EE_NODATA)
        ]

        return df

    def fc2df(self, featurecollection, columns=None):
        df = pd.DataFrame()
        fc_exists = False
        try:
            _fc_exists = featurecollection.first().getInfo()
            fc_exists = True
        except ee.ee_exception.EEException as e:
            pass

        if fc_exists:
            tempfile = str(uuid.uuid4())
            blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
            task_id = self.export_fc_cloudstorage(
                featurecollection, self.BUCKET, blob, "CSV", columns
            )
            self.wait()
            csv = self._download_from_cloudstorage(f"{blob}.csv", f"{tempfile}.csv")
            self.fc_csvs.append((f"{tempfile}.csv", None))

            # uncomment to export shp for QA
            # shp_task_id = self.export_fc_cloudstorage(
            #     featurecollection, self.BUCKET, blob, "SHP", columns
            # )

            df = pd.read_csv(csv, encoding="utf-8")
            self._remove_from_cloudstorage(f"{blob}.csv")
        return df

    def df2fc(
        self, df: pd.DataFrame, geofield: str = "geom"
    ) -> Optional[ee.FeatureCollection]:
        tempfile = str(uuid.uuid4())
        blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
        if df.empty:
            return None

        df.replace(np.inf, 0, inplace=True)
        df.to_csv(f"{tempfile}.csv", encoding="utf-8")
        self._upload_to_cloudstorage(f"{tempfile}.csv", f"{blob}.csv")
        table_asset_name, table_asset_id = self._prep_asset_id(f"scratch/{tempfile}")
        task_id = self._cp_storage_to_ee_table(
            f"gs://{self.BUCKET}/{blob}.csv", table_asset_id, geofield
        )
        self.wait()
        self._remove_from_cloudstorage(f"{blob}.csv")
        self.fc_csvs.append((f"{tempfile}.csv", table_asset_id))
        return ee.FeatureCollection(table_asset_id)

    @property
    def user_grids(self):
        if len(self._grids) < 1:
            gridnames = set(
                self.df_adhoc["GridName"].unique().tolist()
                + self.df_cameratrap_dep["GridName"].unique().tolist()
                + self.df_signsurvey["GridName"].unique().tolist()
            )
            for gridname in gridnames:
                gridcells_query = (
                    f"SELECT CI_GridCellCode, Geom.STAsText() AS geom "
                    f"FROM CI_GridCell gc "
                    f"INNER JOIN CI_Grid g ON (gc.CI_GridID = g.CI_GridID) "
                    f"WHERE g.CI_GridName = '{gridname}' "
                    f"ORDER BY CI_GridCellCode"
                )
                df_gridcells = pd.read_sql(gridcells_query, self.obsconn)
                gridcells_list = df_gridcells.values.tolist()
                self._grids[gridname] = [
                    (wkt.loads(g[1]), {self.CELL_LABEL: g[0]}) for g in gridcells_list
                ]
        return self._grids

    @property
    def zone_ids(self):
        if len(self._zone_ids) < 1:
            self._zone_ids = (
                self.zones.aggregate_histogram(self.ZONES_LABEL).keys().getInfo()
            )
        return self._zone_ids

    @property
    def df_adhoc(self):
        if self._df_adhoc is None:
            _csvpath = "adhoc.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_adhoc = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
                )
            else:
                query = (
                    f"SELECT * FROM dbo.vw_CI_AdHocObservation "
                    f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                    f"AND ObservationDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_adhoc = self._get_df(query)
                print("zonify adhoc")
                self._df_adhoc = self.zonify(self._df_adhoc)
                self._df_adhoc.set_index(self.MASTER_CELL_LABEL, inplace=True)

                if self.save_cache and not self._df_adhoc.empty:
                    self._df_adhoc.to_csv(_csvpath, encoding="utf-8")

        return self._df_adhoc[
            self._df_adhoc[self.MASTER_GRID_LABEL].astype(str) == self.zone
        ]

    @property
    def df_cameratrap(self):
        if self._df_ct is None:
            _csvpath = "cameratrap.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_ct = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col="CameraTrapDeploymentID"
                )
            else:
                query = (
                    f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
                    f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                    f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
                )
				# TODO: modify DB query to only select unique observations for each CameraTrapDeploymentID AND ObservationDateTime (Kim 1)
                _df_ct_dep = self._get_df(query)
                print("zonify camera trap deployments")
                _df_ct_dep = self.zonify(_df_ct_dep)
                _df_ct_dep.set_index("CameraTrapDeploymentID", inplace=True)

                query = (
                    f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                    f"WHERE DATEDIFF(YEAR, ObservationDateTime, '{self.taskdate}') <= "
                    f"{self.inputs['obs_ct']['maxage']} "
                    f"AND ObservationDateTime <= Cast('{self.taskdate}' AS datetime) "
                )
                _df_ct_obs = self._get_df(query)


                _df_ct_obs.set_index("CameraTrapDeploymentID", inplace=True)
                _df_ct_obs["detections"] = (
                    _df_ct_obs["AdultMaleCount"]
                    + _df_ct_obs["AdultFemaleCount"]
                    + _df_ct_obs["AdultSexUnknownCount"]
                    + _df_ct_obs["SubAdultCount"]
                    + _df_ct_obs["YoungCount"]
                )

                self._df_ct = pd.merge(
                    left=_df_ct_dep, right=_df_ct_obs, left_index=True, right_index=True, how='left'
                )
                self._df_ct['detections'].fillna(0, inplace=True)
                if self.save_cache and not self._df_ct.empty:
                    self._df_ct.to_csv(_csvpath, encoding="utf-8")

        return self._df_ct[self._df_ct[self.MASTER_GRID_LABEL].astype(str) == self.zone]

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            _csvpath = "signsurvey.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_ss = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
                )
            else:
                query = (
                    f"SELECT * FROM dbo.vw_CI_SignSurveyObservation "
                    f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                    f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_ss = self._get_df(query)
                print("zonify sign survey")
                self._df_ss = self.zonify(self._df_ss)
                self._df_ss.set_index(self.MASTER_CELL_LABEL, inplace=True)
                # TODO: make sure each data frame has covariates by joining with cov df, check after real data as a check
                # to make sure all cells have covariate data, each dataframe check (Jamie 1)

                if self.save_cache and not self._df_ss.empty:
                    self._df_ss.to_csv(_csvpath, encoding="utf-8")

        return self._df_ss[self._df_ss[self.MASTER_GRID_LABEL].astype(str) == self.zone]

    # def tri(self, dem, scale):
    #     neighbors = dem.neighborhoodToBands(ee.Kernel.square(1.5))
    #     diff = dem.subtract(neighbors)
    #     sq = diff.multiply(diff)
    #     tri = sq.reduce("sum").sqrt().reproject(self.crs, None, scale)
    #     return tri

    # Probably we need more sophisticated covariate definitions (mode of rounded cell vals?)
    # or to sample using smaller gridcell geometries
    @property
    def df_covars(self):
        if self._df_covars is None:
            _csvpath = "covars.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_covars = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
                )
            else:
                sh_ic = ee.ImageCollection(self.inputs["structural_habitat"]["ee_path"])
                hii_ic = ee.ImageCollection(self.inputs["hii"]["ee_path"])
                tri = ee.Image(self.inputs["tri"]["ee_path"])
                # TODO: when we have OSM, point to fc dir and implement get_most_recent_featurecollection (Kim 1)
                roads = ee.FeatureCollection(self.inputs["roads"]["ee_path"])

                structural_habitat, sh_date = self.get_most_recent_image(sh_ic)
                hii, hii_date = self.get_most_recent_image(hii_ic)
                distance_to_roads = roads.distance().clipToCollection(
                    ee.FeatureCollection(self.zones.geometry())
                )

                if structural_habitat and hii:
                    covariates_bands = (
                        structural_habitat.rename("structural_habitat")
                        .unmask(0)
                        .clipToCollection(ee.FeatureCollection(self.zones.geometry()))
                        .addBands(hii.rename("hii"))
                        .addBands(tri.rename("tri"))
                        .addBands(distance_to_roads.rename("distance_to_roads"))
                    )
                    covariates_fc = covariates_bands.reduceRegions(
                        collection=self.gridcells,
                        reducer=ee.Reducer.mean(),
                        scale=self.scale,
                        crs=self.crs,
                        # tileScale=16,  # this causes an ee computation timeout error
                    )
                    self._df_covars = self.fc2df(covariates_fc)

                    if self._df_covars.empty:
                        self._df_covars = pd.DataFrame(
                            columns=[
                                self.MASTER_GRID_LABEL,
                                self.MASTER_CELL_LABEL,
                                "structural_habitat",
                                "hii",
                                "tri",
                                "distance_to_roads",
                            ]
                        )
                    else:
                        self._df_covars.rename(
                            {
                                "zone": self.MASTER_GRID_LABEL,
                                "id": self.MASTER_CELL_LABEL,
                            },
                            axis=1,
                            inplace=True,
                        )
                        covar_stats = self._df_covars.describe()

                        for col in covar_stats.columns:
                             if col=="structural_habitat" or col=="hii" or col=="tri" or col=="distance_to_roads":
                                 self._df_covars[col] = (
                                     self._df_covars[col] - covar_stats[col]["mean"]
                                 ) / covar_stats[col]["std"]

                    self._df_covars = self._df_covars.dropna()
                    self._df_covars.set_index(self.MASTER_CELL_LABEL, inplace=True)

                    if self.save_cache and not self._df_covars.empty:
                        self._df_covars.to_csv(_csvpath, encoding="utf-8")
                else:
                    return None

        return self._df_covars[
            self._df_covars[self.MASTER_GRID_LABEL].astype(str) == self.zone
        ]

    def pbso_integrated(self):
        """Overall function for optimizing function.

        self.presence_covars: matrix with data for covariates that might affect tiger presence
        self.po_detection_covars: matrix with data for covariates that might bias presence-only data
        self.Npsign: single value sign survey
        self.NpCT: single value camera trap

        Returns dataframe of coefficients dataframe (parameter name, value, standard error),
        convergence, message for optimization, and value of negative log-likelihood"""

        beta_names = list(self.presence_covars)
        beta_names[0] = "beta0"
        alpha_names = []
        psign_names = []
        pcam_names = []
        if not self.df_adhoc.empty:
            alpha_names = list(self.po_detection_covars)
            alpha_names[0] = "alpha0"
        if not self.df_signsurvey.empty:
            psign_names = [f"p_sign_{i}" for i in range(0, self.Npsign)]
        if not self.df_cameratrap.empty:
            pcam_names = [f"p_cam_{i}" for i in range(0, self.NpCT)]
        param_names = beta_names + alpha_names + psign_names + pcam_names

        param_guess = np.zeros(len(param_names))

        fit_pbso = minimize(
            self.neg_log_likelihood_int, param_guess, method="BFGS", options={"gtol": 1}
        )
        se_pbso = np.zeros(len(fit_pbso.x))
        # TODO: Output Standard Error of parameter estimates when convergence occurs, catch errors (Jamie 1)
        # Jamie will catch errors, Kim will handle what to do afterwards
        if fit_pbso.success==True:
            se_pbso = np.sqrt(np.diag(fit_pbso.hess_inv))
        tmp = {
            "Parameter name": param_names,
            "Value": fit_pbso.x,
            "Standard error": se_pbso[0],
        }
        # TODO: continue improving variable readability... (Jamie 3)
        p = {
            "coefs": pd.DataFrame(
                tmp, columns=["Parameter name", "Value", "Standard error"]
            ),
            "convergence": fit_pbso.success,
            "optim_message": fit_pbso.message,
            "value": fit_pbso.fun,
        }
        return p

    def neg_log_likelihood_int(self, par):
        """Calculates the negative log-likelihood of the function.
         Par: array list of parameters to optimize
         Returns single value of negative log-likelihood of function"""

        beta = par[0 : self.Nx]
        known_ct = []
        known_sign = []
        lambda0 = np.exp(np.dot(np.array(self.presence_covars), beta))
        self.psi = 1.0 - np.exp(-lambda0)
        nll_po = 0

        zeta = np.empty((len(self.psi), 2))
        zeta[:, 0] = 1.0 - self.psi

        try:
            zeta[:, 1] = np.log(self.psi)
        except RuntimeWarning as e:
            print("No worries, keep going: ",e)

        self.df_zeta = pd.DataFrame(
            {"zeta0": zeta[:, 0], "zeta1": zeta[:, 1]},
            index=self.presence_covars.index.copy(),
        )

        # iterate over unique cameratrap observation IDs, if there are camera trap data
        if not self.df_cameratrap.empty:
            p_cam = expit(
                par[
                    self.Nx
                    + self.Nw
                    + self.Npsign : self.Nx
                    + self.Nw
                    + self.Npsign
                    + self.NpCT
                ]
            )

            # TODO: This seems ugly. See query refactoring note above; or, if we want to preserve the join within
            #  pandas, maybe we can specify on the join using a class constant.
            uniqueid_x = f"{self.UNIQUE_ID_LABEL}_x"
            # ct_ids is a list of CT obs unique ids, not deployment ids
            ct_ids = list(self.df_cameratrap[uniqueid_x].unique())
            for i in ct_ids:
                try:
                    self.df_zeta.loc[
                        self.df_cameratrap[self.df_cameratrap[uniqueid_x] == i][
                            self.MASTER_CELL_LABEL  
                        ].values[0],
                        "zeta1",
                    ] += (
                        self.df_cameratrap[self.df_cameratrap[uniqueid_x] == i][
                            "detections"
                        ].values[0]
                    ) * np.log(
                        p_cam[self.NpCT - 1]
                    ) + (
                        self.df_cameratrap[self.df_cameratrap[uniqueid_x] == i][
                            "days"
                        ].values[0]
                        - self.df_cameratrap[self.df_cameratrap[uniqueid_x] == i][
                            "detections"
                        ].values[0]
                    ) * np.log(
                        1.0 - p_cam[self.NpCT - 1]
                    )
                except KeyError as e:
                    # pass
                    print(f"df_zeta has no row with {self.MASTER_CELL_LABEL} = {e}")

            known_ct = self.df_cameratrap[self.df_cameratrap["detections"] > 0][
                self.MASTER_CELL_LABEL  
            ].tolist()

        # iterate over unique set of surveys, if there are sign survey data
        if not self.df_signsurvey.empty:
            p_sign = expit(par[self.Nx + self.Nw : self.Nx + self.Nw + self.Npsign])

            survey_ids = list(self.df_signsurvey[self.UNIQUE_ID_LABEL].unique())
            for j in survey_ids:
                self.df_zeta.loc[
                    self.df_signsurvey.index[
                        (self.df_signsurvey[self.UNIQUE_ID_LABEL] == j)
                    ].tolist()[0],
                    "zeta1",
                ] += (
                    self.df_signsurvey[self.df_signsurvey[self.UNIQUE_ID_LABEL] == j][
                        "detections"
                    ].values[0]
                ) * np.log(
                    p_sign[self.Npsign - 1]
                ) + (
                    self.df_signsurvey[self.df_signsurvey[self.UNIQUE_ID_LABEL] == j][
                        "NumberOfReplicatesSurveyed"
                    ].values[0]
                    - self.df_signsurvey[self.df_signsurvey[self.UNIQUE_ID_LABEL] == j][
                        "detections"
                    ].values[0]
                ) * np.log(
                    1.0 - p_sign[self.Npsign - 1]
                )

            known_sign = self.df_signsurvey.index[
                (self.df_signsurvey["detections"] > 0)
            ].tolist()

        known_occurrences = list(set(known_sign + known_ct))
        self.df_zeta.loc[known_occurrences, "zeta0"] = 0

        self.df_zeta["lik_so"] = self.df_zeta.loc[:, "zeta1"]
        self.df_zeta.loc[
            self.df_zeta.index[(self.df_zeta["zeta0"] != 0)].tolist(), "lik_so"
        ] += np.log(
            self.df_zeta.loc[
                self.df_zeta.index[(self.df_zeta["zeta0"] != 0)].tolist(), "zeta0"
            ]
        )
        self.df_zeta["lambda0"] = lambda0

        # TODO: handle ad hoc with density>0 (Jamie 1), fields "density" and "standard_error"
        if not self.df_adhoc.empty:
            alpha = par[self.Nx : self.Nx + self.Nw]
            tw = np.dot(np.array(self.po_detection_covars), alpha)
            p_thin = expit(tw)
            self.df_zeta["pthin"] = p_thin
            adhoc_indices = list(
                set(self.df_adhoc.index.values) & set(self.df_zeta.index.values)
            )
            nll_po = -1.0 * (
                (-1.0 * sum(lambda0 * p_thin))
                + sum(
                    np.log(
                        self.df_zeta.loc[adhoc_indices, "lambda0"]
                        * self.df_zeta.loc[adhoc_indices, "pthin"]
                    )
                )
            )

        nll_so = -1.0 * sum(self.df_zeta["lik_so"])

        # TODO: enhance prob calcs to incorporate observation age
        return nll_po + nll_so

    def predict_surface(self):
        """Create predicted probability surface for each grid cell.
         Par: list of parameter values that have been optimized to convert to probability surface
         Returns data frame indexed by grid cell code with predicted probability surface for each grid cell
         and a ratio of conditional psi to unconditional psi"""

        # predicted probability surface
        self.df_zeta["cond_psi"] = (np.exp(self.df_zeta.loc[:, "zeta1"])) / (
            self.df_zeta.loc[:, "zeta0"] + np.exp(self.df_zeta.loc[:, "zeta1"])
        )
        # ratio of conditional psi to unconditional psi, incorporates sampling effort
        self.df_zeta["ratio_psi"] = self.psi / self.df_zeta.loc[:, "zeta0"]
        df_predictsurface = self.df_zeta.loc[:, ["cond_psi", "ratio_psi"]]

        return df_predictsurface

    def calc(self):
        # prob_images = []
        for zone in self.zone_ids:
            self.zone = zone  # all dataframes are filtered by this
            self.Nx = 0
            self.Nw = 0
            # TODO: set these dynamically; right now assumes constant detection probability for SS/CT data
            self.Npsign = 1
            self.NpCT = 1

            if self.df_adhoc.empty:
                print(
                    f"No adhoc data observations for zone {self.zone} "
                    f"in the {self.inputs['obs_adhoc']['maxage']} years prior to {self.taskdate}."
                )
            if self.df_signsurvey.empty:
                print(
                    f"No sign survey data observations for zone {self.zone} "
                    f"in the {self.inputs['obs_ss']['maxage']} years prior to {self.taskdate}."
                )
                self.Npsign = 0
            if self.df_cameratrap.empty:
                print(
                    f"No camera trap data observations for zone {self.zone} "
                    f"in the {self.inputs['obs_ct']['maxage']} years prior to {self.taskdate}."
                )
                self.NpCT = 0

            # We need observations from at least one observation type per zone
            if self.df_signsurvey.empty and self.df_cameratrap.empty:
                print(f"No structured data for zone {self.zone}")
                continue
            # TODO: ensure we have at least some structured data for every zone, and then uncomment these lines and
            #  remove the preceding two, so that task fails if no structured data for ANY zone.
            #     self.status = self.FAILED
            #     raise NotImplementedError("Probability calculation without structured data is not defined.")

            self.po_detection_covars = self.df_covars[["tri", "distance_to_roads"]]
            # TODO: Can 'alpha' and 'beta' be added to these dfs here?
            self.po_detection_covars.insert(0, "Int", 1)
            self.presence_covars = self.df_covars[["structural_habitat", "hii"]]
            self.presence_covars.insert(0, "Int", 1)
            self.Nx = self.presence_covars.shape[1]
            if not self.df_adhoc.empty:
                self.Nw = self.po_detection_covars.shape[1]

            # TODO: set class properties instead of returning
            m = self.pbso_integrated()
            print(m)
            probs = self.predict_surface()
            print(probs)
            probs.to_csv(f"probs{self.zone}.csv")
            # probs = pd.read_csv(
            #     f"probs{self.zone}.csv",
            #     encoding="utf-8",
            #     index_col=self.MASTER_CELL_LABEL,
            # )

            df_prob = pd.merge(
                left=probs, right=self.df_covars, left_index=True, right_index=True
            ).loc[:, ["cond_psi", "ratio_psi", ".geo"]]
            df_prob.rename(
                columns={".geo": "geom"}, inplace=True
            )  # ee cannot handle geom label starting with '.'
            df_prob["geom"] = df_prob["geom"].apply(lambda x: wkt.dumps(json.loads(x)))
            fc_prob = self.df2fc(df_prob)

            probzone = fc_prob.reduceToImage(["cond_psi"], ee.Reducer.max()).rename(
                "probability"
            )
            self.export_image_ee(probzone, f"probability{self.zone}")
            effortzone = fc_prob.reduceToImage(["ratio_psi"], ee.Reducer.max()).rename(
                "effort"
            )
            self.export_image_ee(effortzone, f"effortzone{self.zone}")
            print(f"Started image exports for zone {self.zone}")

            # "Fake" probability used for 6/17/20 calcs -- not for production use
            # probcells = []
            # for cell in self.grids[gridname]:
            #     gridcellcode = cell[1][self.cell_label]
            #     detections = 0
            #     try:
            #         detections = int(
            #             df_signsurvey[
            #                 df_signsurvey[self.cell_label].str.match(gridcellcode)
            #             ]["detections"].sum()
            #         )
            #         if detections > 1:
            #             detections = 1
            #     except KeyError:
            #         pass
            #
            #     props = cell[1]
            #     props["probability"] = detections
            #     probcell = ee.Feature(cell[0], props)
            #     probcells.append(probcell)
            #
            # fake_prob = (
            #     ee.FeatureCollection(probcells)
            #     .reduceToImage(["probability"], ee.Reducer.max())
            #     .rename("probability")
            # )
            # self.export_image_ee(fake_prob, "hab/probability")

        # TODO: add (? or otherwise combine) all probability images, one for each grid (Kim, TBD)
        # self.export_image_ee(combined_images, "hab/probability")

    def check_inputs(self):
        super().check_inputs()

    def clean_up(self, **kwargs):
        if self.status == self.FAILED:
            return

        if self.fc_csvs:
            for csv, table_asset_id in self.fc_csvs:
                if csv and Path(csv).exists():
                    Path(csv).unlink()
                if table_asset_id:
                    try:
                        asset = ee.data.getAsset(table_asset_id)
                        ee.data.deleteAsset(table_asset_id)
                    except ee.ee_exception.EEException:
                        print(f"{table_asset_id} does not exist; skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate", default=datetime.now(timezone.utc).date())
    parser.add_argument("-s", "--species", default="Panthera_tigris")
    parser.add_argument("--scenario", default=SCLTask.CANONICAL)
    options = parser.parse_args()
    sclprobcoeff_task = SCLProbabilityCoefficients(**vars(options))
    sclprobcoeff_task.run()
