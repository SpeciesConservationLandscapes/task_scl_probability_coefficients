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
        self._df_ct = None
        self._df_ss = None
        self._df_covars = None
        self.zone = None
        self.Nx = 0
        self.Nw = 0
        # TODO: set these dynamically, right now assumes constant detection probability for sign survey and camera
        #  trap data
        self.Npsign = 1
        self.NpCT = 1
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

    # TODO: modify DB query to only select unique observations for each CameraTrapDeploymentID AND ObservationDateTime
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
            self._df_ct_obs["detections"] = (
                self._df_ct_obs["AdultMaleCount"]
                + self._df_ct_obs["AdultFemaleCount"]
                + self._df_ct_obs["AdultSexUnknownCount"]
                + self._df_ct_obs["SubAdultCount"]
                + self._df_ct_obs["YoungCount"]
            )

        return self._df_ct_obs

    @property
    def df_cameratrap(self):
        if self._df_ct is None:
            self._df_ct = pd.merge(
                left=self.df_cameratrap_dep,
                right=self.df_cameratrap_obs,
                left_index=True,
                right_index=True,
            )
        return self._df_ct

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
        alpha_names = []
        psign_names = []
        pcam_names = []
        if not self.df_adhoc.empty:
            # TODO: set self.po_detection_covars and any other properties from __init__ so that code can run and
            #  produce empty results (here we'd get `TypeError: 'NoneType' object is not iterable`)
            alpha_names = list(self.po_detection_covars)
            alpha_names[0] = "alpha0"
        if not self.df_signsurvey.empty:
            psign_names = [f"p_sign_{i}" for i in range(0, self.Npsign)]
        if not self.df_cameratrap.empty:
            pcam_names = [f"p_cam_{i}" for i in range(0, self.NpCT)]
        param_names = beta_names + alpha_names + psign_names + pcam_names

        # TODO: Should be able to remove the lines above and just get param_names from
        #  self.presence_covar.columns + self.po_detection_covars.columns
        #  assuming those dfs are formatted correctly (see note in calc())
        param_guess = np.zeros(len(param_names))
        # TODO: Make sure convergence, a different method might be needed (can be tested outside of task)
        fit_pbso = minimize(
            self.neg_log_likelihood_int,
            param_guess,
            method="BFGS",
            options={"gtol": 1e-08},
        )
        se_pbso = np.zeros(len(fit_pbso.x))
        # TODO: Output Standard Error of parameter estimates when convergence occurs
        # if fit_pbso.success==True:
        #    se_pbso = np.sqrt(np.diag(fit_pbso.hess_inv))
        tmp = {
            "Parameter name": param_names,
            "Value": fit_pbso.x,
            "Standard error": se_pbso[0],
        }
        # TODO: continue improving variable readability...
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
    # TODO: make sure neg_log_likelihood can account for no adhoc data
    def neg_log_likelihood_int(self, par):
        """Calculates the negative log-likelihood of the function.
         Par: array list of parameters to optimize
         Returns single value of negative log-likelihood of function"""

        beta = par[0 : self.Nx]
        known_ct = []
        known_sign = []
        lambda0 = np.exp(np.dot(np.array(self.presence_covars), beta))
        # TODO: This should get initialized in __init__ so that predict_surface won't fail
        self.psi = 1.0 - np.exp(-lambda0)
        nll_po = 0

        zeta = np.empty((len(self.psi), 2))
        zeta[:, 0] = 1.0 - self.psi
        # TODO: handle RuntimeWarning divide by zero with log
        zeta[:, 1] = np.log(self.psi)
        # TODO: same __init__ issue
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

            ct_ids = list(self.df_cameratrap.UniqueID_y.unique())
            for i in ct_ids:
                try:
                    self.df_zeta.loc[
                        self.df_cameratrap[self.df_cameratrap["UniqueID_y"] == i][
                            "GridCellCode"
                        ].values[0],
                        "zeta1",
                    ] += (
                        self.df_cameratrap[self.df_cameratrap["UniqueID_y"] == i][
                            "detections"
                        ].values[0]
                    ) * np.log(
                        p_cam[self.NpCT - 1]
                    ) + (
                        self.df_cameratrap[self.df_cameratrap["UniqueID_y"] == i][
                            "days"
                        ].values[0]
                        - self.df_cameratrap[self.df_cameratrap["UniqueID_y"] == i][
                            "detections"
                        ].values[0]
                    ) * np.log(
                        1.0 - p_cam[self.NpCT - 1]
                    )
                except KeyError:
                    print("missing camera trap grid cell")

            known_ct = self.df_cameratrap[self.df_cameratrap["detections"] > 0][
                "GridCellCode"
            ].tolist()

        # iterate over unique set of surveys, if there are sign survey data
        if not self.df_signsurvey.empty:
            p_sign = expit(par[self.Nx + self.Nw : self.Nx + self.Nw + self.Npsign])

            survey_ids = list(self.df_signsurvey.UniqueID.unique())
            for j in survey_ids:
                self.df_zeta.loc[
                    self.df_signsurvey.index[
                        (self.df_signsurvey["UniqueID"] == j)
                    ].tolist()[0],
                    "zeta1",
                ] += (
                    self.df_signsurvey[self.df_signsurvey["UniqueID"] == j][
                        "detections"
                    ].values[0]
                ) * np.log(
                    p_sign[self.Npsign - 1]
                ) + (
                    self.df_signsurvey[self.df_signsurvey["UniqueID"] == j][
                        "NumberOfReplicatesSurveyed"
                    ].values[0]
                    - self.df_signsurvey[self.df_signsurvey["UniqueID"] == j][
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

        if not self.df_adhoc.empty:
            alpha = par[self.Nx : self.Nx + self.Nw]
            tw = np.dot(np.array(self.po_detection_covars), alpha)
            p_thin = expit(tw)
            self.df_zeta["pthin"] = p_thin
            # TODO: adhoc_indices is not used -- remove, or is something else missing?
            adhoc_indices = list(
                set(self.df_adhoc.index.values) & set(self.df_zeta.index.values)
            )

        nll_so = -1.0 * sum(self.df_zeta["lik_so"])

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
        # print(self.zone_ids)
        # prob_images = []
        for zone in self.zone_ids:
            self.zone = zone  # all dataframes are filtered by this
            # output empty dataframes to user, modify sign survey and camera trap number of parameters
            if self.df_adhoc.empty:
                print(
                    f"There are no adhoc data observations for grid {gridname} during this time period."
                )
            if self.df_signsurvey.empty:
                print(
                    f"There are no sign survey data observations for grid {gridname} during this time period."
                )
                self.Npsign = 0
            if self.df_cameratrap.empty:
                print(
                    f"There are no camera trap data observations for grid {gridname} during this time period."
                )
                self.NpCT = 0

            # print(self.df_adhoc)
            # print(self.df_signsurvey)
            # print(self.df_cameratrap_dep)
            # print(self.df_cameratrap_obs)
            # print(self.df_cameratrap)
            self.df_cameratrap.to_csv("ct.csv", encoding="utf-8")
            self.df_adhoc.to_csv("adhoc.csv", encoding="utf-8")
            self.df_signsurvey.to_csv("signsurvey.csv", encoding="utf-8")

            # print(self.df_covars)
            # self.df_covars.to_csv("covars.csv", encoding="utf-8")
            # self.df_covars = pd.read_csv(
            #     "covars.csv", encoding="utf-8", index_col=self.cell_label
            # )

            self.po_detection_covars = df_covars[["tri", "distance_to_roads"]]
            # TODO: Can 'alpha' and 'beta' be added to these dfs here?
            self.po_detection_covars.insert(0, "Int", 1)
            self.presence_covars = df_covars[["structural_habitat", "hii"]]
            self.presence_covars.insert(0, "Int", 1)
            self.Nx = self.presence_covars.shape[1]
            if not self.df_adhoc.empty:
                self.Nw = self.po_detection_covars.shape[1]
            else:
                self.Nw = 0

            # TODO: set class properties instead of returning
            m = self.pbso_integrated()
            print(m)
            probs = self.predict_surface()
            print(probs)

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
