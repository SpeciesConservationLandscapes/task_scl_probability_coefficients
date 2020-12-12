import argparse
import ee
import os
import numpy as np
import pandas as pd
import pyodbc
import time
import uuid
from datetime import datetime, timezone
from geomet import wkt
from google.cloud.storage import Client
from google.cloud.exceptions import NotFound
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from task_base import SCLTask


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
    inputs = {
        "obs_adhoc": {"maxage": 6},
        "obs_ss": {"maxage": 6},
        "obs_ct": {"maxage": 6},
        "hii": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "projects/HII/v1/hii",
            "maxage": 30,
        },
        "dem": {"ee_type": SCLTask.IMAGE, "ee_path": "CGIAR/SRTM90_V4", "static": True},
        # TODO: replace with roads from OSM and calculate distance
        "roads": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/Panthera-Earth-Engine/Roads/SouthAsiaRoads",
            "maxage": 1,
        },
        "structural_habitat": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/structural_habitat",
            "maxage": 10,  # until we have full-range SH for every year
        },
        "zones": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/zones",
            "static": True,
        },
        "gridcells": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/gridcells",
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
        self._gridname = None
        self._df_adhoc = None
        self._df_ct_dep = None
        self._df_ct_obs = None
        self._df_ss = None
        self._df_covars = None
        self.zone = None
        self.Nx = 0
        self.Nw = 0
        self.Npsign = 0
        self.NpCT = 0
        # coefficients relevant to presence-only and background detection only
        self.po_detection_covars = None
        # coefficients relevant to occupancy, shared across models
        self.presence_covars = None

        self.zones = ee.FeatureCollection(self.inputs["zones"]["ee_path"])
        self.gridcells = ee.FeatureCollection(self.inputs["gridcells"]["ee_path"])
        self.fc_csvs = []

    def _download_from_cloudstorage(self, blob_path: str, local_path: str) -> str:
        client = Client()
        bucket = client.get_bucket(self.BUCKET)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return local_path

    def _remove_from_cloudstorage(self, blob_path: str):
        client = Client()
        bucket = client.bucket(self.BUCKET)
        try:  # don't fail entire task if this fails
            bucket.delete_blob(blob_path)
        except NotFound:
            print(f"{blob_path} not found")

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
        # matching_zones = self.zones.filterBounds(centroid)
        intersects = ee.Filter.intersects(".geo", None, ".geo")
        matching_zones = ee.Join.simple().apply(self.zones, obs_feature, intersects)
        zone_id_true = ee.Number(matching_zones.first().get(self.ZONES_LABEL))
        zone_id_false = ee.Number(self.EE_NODATA)
        zone_id = ee.Number(
            ee.Algorithms.If(matching_zones.size().gte(1), zone_id_true, zone_id_false)
        )

        gridcell_id_true = ee.Number(
            self.gridcells.filter(ee.Filter.eq("zone", zone_id))
            .filterBounds(centroid)
            .first()
            .get(self.MASTER_CELL_ID_LABEL)
        )
        gridcell_id_false = ee.Number(self.EE_NODATA)
        gridcell_id = ee.Number(
            ee.Algorithms.If(
                zone_id.neq(self.EE_NODATA), gridcell_id_true, gridcell_id_false
            )
        )

        obs_feature = obs_feature.setMulti(
            {self.MASTER_GRID_LABEL: zone_id, self.MASTER_CELL_LABEL: gridcell_id}
        )

        return obs_feature

    # add "master grid" and "master gridcell" to df
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
        fcsize = featurecollection.size().getInfo()

        if fcsize > 0:
            tempfile = str(uuid.uuid4())
            blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
            task_id = self.export_fc_cloudstorage(
                featurecollection, self.BUCKET, blob, "CSV", columns
            )
            self.wait()
            csv = self._download_from_cloudstorage(f"{blob}.csv", f"{tempfile}.csv")
            self.fc_csvs.append(csv)

            # uncomment to export shp for QA
            # shp_task_id = self.export_fc_cloudstorage(
            #     featurecollection, self.BUCKET, blob, "SHP", columns
            # )

            df = pd.read_csv(csv, encoding="utf-8")
            self._remove_from_cloudstorage(f"{blob}.csv")
        return df

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
            query = (
                f"SELECT * FROM dbo.vw_CI_AdHocObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                f"AND ObservationDate <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_adhoc = self._get_df(query)
            self._df_adhoc = self.zonify(self._df_adhoc)
            self._df_adhoc.set_index(self.MASTER_CELL_LABEL, inplace=True)
        return self._df_adhoc[
            self._df_adhoc[self.MASTER_GRID_LABEL].astype(str) == self.zone
        ]

    # TODO: refactor these CT dfs once we figure out new schema (use adhoc/ss as recipe)
    @property
    def df_cameratrap_dep(self):
        if self._df_ct_dep is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
                f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_ct_dep = self._get_df(query)
            self._df_ct_dep = self.zonify(self._df_ct_dep)
            self._df_ct_dep.set_index("CameraTrapDeploymentID", inplace=True)
        return self._df_ct_dep[
            self._df_ct_dep[self.MASTER_GRID_LABEL].astype(str) == self.zone
        ]

    # This will be the same across all zones, because there is no way to look up master grids based on location.
    # The assumption is that this will be joined to self.df_cameratrap_dep, which is zone-filtered.
    @property
    def df_cameratrap_obs(self):
        if self._df_ct_obs is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDateTime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND ObservationDateTime <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_ct_obs = self._get_df(query)
            self._df_ct_obs.set_index("CameraTrapDeploymentID", inplace=True)
        return self._df_ct_obs

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_SignSurveyObservation "
                f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_ss = self._get_df(query)
            self._df_ss = self.zonify(self._df_ss)
            self._df_ss.set_index(self.MASTER_CELL_LABEL, inplace=True)
        return self._df_ss[self._df_ss[self.MASTER_GRID_LABEL].astype(str) == self.zone]

    def tri(self, dem, scale):
        neighbors = dem.neighborhoodToBands(ee.Kernel.square(1.5))
        diff = dem.subtract(neighbors)
        sq = diff.multiply(diff)
        tri = sq.reduce("sum").sqrt().reproject(self.crs, None, scale)
        return tri

    # Probably we need more sophisticated covariate definitions (mode of rounded cell vals?)
    # or to sample using smaller gridcell geometries
    @property
    def df_covars(self):
        if self._df_covars is None:
            sh_ic = ee.ImageCollection(self.inputs["structural_habitat"]["ee_path"])
            hii_ic = ee.ImageCollection(self.inputs["hii"]["ee_path"])
            dem = ee.Image(self.inputs["dem"]["ee_path"])
            # TODO: when we have OSM, point to fc dir and implement get_most_recent_featurecollection
            roads = ee.FeatureCollection(self.inputs["roads"]["ee_path"])

            structural_habitat, sh_date = self.get_most_recent_image(sh_ic)
            hii, hii_date = self.get_most_recent_image(hii_ic)
            tri = self.tri(dem, 90)
            distance_to_roads = roads.distance().clipToCollection(self.zones)

            if structural_habitat and hii:
                covariates_bands = (
                    structural_habitat.rename("structural_habitat")
                    .addBands(hii.rename("hii"))
                    .addBands(tri.rename("tri"))
                    .addBands(distance_to_roads.rename("distance_to_roads"))
                )
                covariates_fc = covariates_bands.reduceRegions(
                    collection=self.gridcells,
                    reducer=ee.Reducer.mean(),
                    scale=self.scale,
                    crs=self.crs,
                )
                self._df_covars = self.fc2df(covariates_fc)

                if self._df_covars.empty:
                    self._df_covars[self.MASTER_GRID_LABEL] = pd.Series(dtype="object")
                    self._df_covars[self.MASTER_CELL_LABEL] = pd.Series(dtype="object")
                else:
                    self._df_covars.rename(
                        {"zone": self.MASTER_GRID_LABEL, "id": self.MASTER_CELL_LABEL},
                        axis=1,
                        inplace=True,
                    )
                    # covar_stats = self._df_covars.describe()
                    # TODO: determine whether we need this anymore
                    #  at master grid level. If we do, need to change the logic for choosing which columns to modify.
                    # for col in covar_stats.columns:
                    #     if not col.startswith("Unnamed"):
                    #         self._df_covars[col] = (
                    #             self._df_covars[col] - covar_stats[col]["mean"]
                    #         ) / covar_stats[col]["std"]

                # TODO: check this -- means no row for any cell with ANY missing covars
                # self._df_covars = self._df_covars.dropna()
                self._df_covars.set_index(self.MASTER_CELL_LABEL, inplace=True)
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
        alpha_names = list(self.po_detection_covars)
        alpha_names[0] = "alpha0"
        psign_names = [f"p_sign_{i}" for i in range(0, self.Npsign)]
        pcam_names = [f"p_cam_{i}" for i in range(0, self.NpCT)]
        param_names = beta_names + alpha_names + psign_names + pcam_names
        # TODO: Should be able to remove the lines above and just get param_names from
        #  self.presence_covar.columns + self.po_detection_covars.columns
        #  assuming those dfs are formatted correctly (see note in calc())
        param_guess = np.zeros(len(param_names))
        fit_pbso = minimize(
            self.neg_log_likelihood_int,
            param_guess,
            method="BFGS",
            options={"gtol": 1e-08},
        )
        se_pbso = np.zeros(len(fit_pbso.x))
        # if fit_pbso.success==True:
        #    se_pbso = np.sqrt(np.diag(fit_pbso.hess_inv))
        tmp = {
            "Parameter name": param_names,
            "Value": fit_pbso.x,
            "Standard error": se_pbso[0],
        }
        p = {
            "coefs": pd.DataFrame(
                tmp, columns=["Parameter name", "Value", "Standard error"]
            ),
            "convergence": fit_pbso.success,
            "optim_message": fit_pbso.message,
            "value": fit_pbso.fun,
        }
        return p

    # TODO: refactor par to use self.df_covars?
    # TODO: replace CT with self.df_* once we've figured out proper schema, and po_data with self.df_adhoc (see below)
    def neg_log_likelihood_int(self, par, CT, po_data):
        """Calculates the negative log-likelihood of the function.
         Par: array list of parameters to optimize
         Returns single value of negative log-likelihood of function"""

        beta = par[0 : self.Nx]
        alpha = par[self.Nx : self.Nx + self.Nw]
        p_sign = expit(par[self.Nx + self.Nw : self.Nx + self.Nw + self.Npsign])
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
        lambda0 = np.exp(np.dot(np.array(self.presence_covars), beta))
        self.df_adhoc["lambda0"] = lambda0
        psi = 1.0 - np.exp(-lambda0)
        tw = np.dot(np.array(self.po_detection_covars), alpha)
        p_thin = expit(tw)
        # TODO: up to this point we're using self.df_adhoc, indexed like all the others by self.cell_label.
        #  But below we use po_data, which is a list of the nth rows in the (sorted) list of grid cell labels
        #  We should refactor out po_data and just use self.df_adhoc, indexed by cell label.
        self.df_adhoc["p_thin"] = p_thin
        zeta = np.empty((len(psi), 2))
        zeta[:, 0] = 1.0 - psi
        zeta[:, 1] = np.log(psi)

        # TODO: refactor out "cell": use self.cell_label
        #  But need to understand CT structure in order to know how to group, either in sql or in df
        #  CT.csv: # of detections (not sum of different kinds, apparently) per -- ?
        #  Should CT start out as np.zeroes?
        for i in range(0, len(CT["det"])):
            zeta[CT["cell"][i] - 1, 1] = (
                zeta[CT["cell"][i] - 1, 1]
                + (CT["det"][i]) * np.log(p_cam[CT["PI"][i] - 1])
                + (CT["days"][i] - CT["det"][i]) * np.log(1.0 - p_cam[CT["PI"][i] - 1])
            )

        # iterate over unique set of surveys
        survey_ids = list(self.df_signsurvey["SignSurveyID"].unique())
        for j in survey_ids:
            zeta[self.df_signsurvey[self.CELL_LABEL][j] - 1, 1] = (
                zeta[self.df_signsurvey[self.CELL_LABEL][j] - 1, 1]
                + (self.df_signsurvey["detections"][j])
                * np.log(p_sign[self.df_signsurvey["SignSurveyID"][j] - 1])
                + (
                    self.df_signsurvey["NumberOfReplicatesSurveyed"][j]
                    - self.df_signsurvey["detections"][j]
                )
                * np.log(1.0 - p_sign[self.df_signsurvey["SignSurveyID"][j] - 1])
            )

        # TODO: make variable names more readable
        one = self.df_signsurvey[self.df_signsurvey["detections"] > 0][self.CELL_LABEL]
        two = CT[CT["det"] > 0][self.CELL_LABEL]
        known_occurrences = list(set(one.append(two)))

        zeta[np.array(known_occurrences) - 1, 0] = 0

        lik_so = []
        for i in range(0, len(zeta[:, 0])):
            if zeta[i, 0] == 0:
                lik_so.append(zeta[i, 1])
            else:
                lik_so.append(np.log(zeta[i, 0]) + zeta[i, 1])

        nll_po = -1.0 * (+sum(np.log(lambda0[po_data - 1] * p_thin[po_data - 1])))
        nll_so = -1.0 * sum(lik_so)

        return nll_po[0] + nll_so

    # TODO: DRY this up to avoid the repeated lines (should look like neg_log_likelihood_int after refactoring params)
    def predict_surface(self, par, CT, df_signsurvey, griddata):
        """Create predicted probability surface for each grid cell.
         Par: list of parameter values that have been optimized to convert to probability surface
         Returns data frame that includes grid code, grid cell number and predicted probability surface for each grid
         cell"""

        par = np.array(par)
        beta = par[0 : self.Nx]
        p_sign = expit(par[self.Nx + self.Nw : self.Nx + self.Nw + self.Npsign])
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
        lambda0 = np.exp(np.dot(np.array(self.presence_covars), beta))
        psi = 1.0 - np.exp(-lambda0)
        zeta = np.empty((len(psi), 2))
        zeta[:, 0] = 1.0 - psi
        zeta[:, 1] = np.log(psi)

        for i in range(0, len(CT["det"])):
            zeta[CT["cell"][i] - 1, 1] = (
                zeta[CT["cell"][i] - 1, 1]
                + (CT["det"][i]) * np.log(p_cam[CT["PI"][i] - 1])
                + (CT["days"][i] - CT["det"][i]) * np.log(1.0 - p_cam[CT["PI"][i] - 1])
            )

        for j in range(0, len(df_signsurvey["detections"])):
            zeta[df_signsurvey["cell"][j] - 1, 1] = (
                zeta[df_signsurvey["cell"][j] - 1, 1]
                + (df_signsurvey["detections"][j])
                * np.log(p_sign[df_signsurvey["SignSurveyID"][j] - 1])
                + (
                    df_signsurvey["NumberOfReplicatesSurveyed"][j]
                    - df_signsurvey["detections"][j]
                )
                * np.log(1.0 - p_sign[df_signsurvey["SignSurveyID"][j] - 1])
            )

        one = df_signsurvey[df_signsurvey["detections"] > 0]["cell"]
        two = CT[CT["det"] > 0]["cell"]
        known_occurrences = list(set(one.append(two)))

        zeta[np.array(known_occurrences) - 1, 0] = 0
        cond_psi = [zeta[i, 1] / sum(zeta[i, :]) for i in range(0, len(psi))]

        cond_prob = 1.0 - (1.0 - np.exp(np.multiply(-1, cond_psi)))
        gridcells = [i for i in range(1, len(psi) + 1)]
        temp = {
            self.CELL_LABEL: griddata[self.CELL_LABEL],
            "gridcells": gridcells,
            "condprob": cond_prob,
        }
        prob_out = pd.DataFrame(
            temp, columns=[self.CELL_LABEL, "gridcells", "condprob"]
        )

        # TODO: return 2 dataframes: 1) conditional, 2) ratio of conditional to unconditional
        return prob_out

    def calc(self):
        # print(self.zone_ids)
        # prob_images = []
        for zone in self.zone_ids:
            self.zone = zone  # all dataframes are filtered by this
            # print(self.df_adhoc)
            # print(self.df_signsurvey)
            # print(self.df_cameratrap_dep)
            # print(self.df_cameratrap_obs)
            # print(self.df_covars)
            # self.df_covars.to_csv("covars.csv", encoding="utf-8")
            # self.df_covars = pd.read_csv(
            #     "covars.csv", encoding="utf-8", index_col=self.cell_label
            # )

            # TODO: set these dynamically
            self.Nx = 3
            self.Nw = 3
            self.Npsign = 1
            self.NpCT = 1

            # self.po_detection_covars = self.df_covars[["tri", "distance_to_roads"]]
            # TODO: Why do we need the extra columns? Can 'alpha' and 'beta' be added to these dfs here?
            # self.po_detection_covars.insert(0, "Int", 1)
            # self.presence_covars = self.df_covars[["structural_habitat", "hii"]]
            # self.presence_covars.insert(0, "Int", 1)

            # m = self.pbso_integrated()
            # probs = self.predict_surface(m["coefs"]["Value"], df_cameratrap_dep, df_signsurvey, df_covars)

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

        # TODO: add (? or otherwise combine) all probability images, one for each grid
        # self.export_image_ee(combined_images, "hab/probability")

    def check_inputs(self):
        super().check_inputs()

    def clean_up(self, **kwargs):
        if self.status == self.FAILED:
            return

        if self.fc_csvs:
            for csv in self.fc_csvs:
                Path(csv).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate", default=datetime.now(timezone.utc).date())
    parser.add_argument("-s", "--species", default="Panthera_tigris")
    parser.add_argument("--scenario", default=SCLTask.CANONICAL)
    options = parser.parse_args()
    sclprobcoeff_task = SCLProbabilityCoefficients(**vars(options))
    sclprobcoeff_task.run()
