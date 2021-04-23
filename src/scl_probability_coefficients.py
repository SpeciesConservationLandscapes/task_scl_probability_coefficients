import os
import argparse
import pyodbc
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import ee
from datetime import datetime, timezone
from task_base import SCLTask
from geomet import wkt


class SCLProbabilityCoefficients(SCLTask):
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
        # TODO: replace with roads from OSM and calculate distance (Kim 1)
        "roads": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/Panthera-Earth-Engine/Roads/SouthAsiaRoads",
            "maxage": 1,
        },
        "structural_habitat": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": f"projects/SCL/v1/Panthera_tigris/structural_habitat",
            "maxage": 10,  # until we have full-range SH for every year
        },
    }
    grid_label = "GridName"
    cell_label = "GridCellCode"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_aoi_from_ee(
            "projects/SCL/v1/Panthera_tigris/sumatra_poc_aoi"
        )  # temporary

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

        self._grids = {}
        self._gridname = None
        self._df_adhoc = None
        self._df_ct_dep = None
        self._df_ct_obs = None
        self._df_ct = None
        self._df_ss = None

        self.Nx = 0
        self.Nw = 0
        self.Npsign = 1
        self.NpCT = 1
        self.po_detection_covars = (
            None
        )  # coefficients relevant to presence-only and background detection only
        self.presence_covars = (
            None
        )  # coefficients relevant to occupancy, shared across models

    def _reset_df_caches(self):
        self._df_adhoc = None
        self._df_ct_dep = None
        self._df_ct_obs = None
        self._df_ct = None
        self._df_ss = None

    def _get_df(self, query, index_field=cell_label):

        _scenario_clause = (
            f"AND ScenarioName IS NULL OR ScenarioName = '{self.CANONICAL}'"
        )
        if self.scenario and self.scenario != self.CANONICAL:
            _scenario_clause = f"AND ScenarioName = '{self.scenario}'"

        query = f"{query}  {_scenario_clause}"
        df = pd.read_sql(query, self.obsconn)
        df.set_index(index_field, inplace=True)
        return df

    @property
    def grids(self):
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
                    (wkt.loads(g[1]), {self.cell_label: g[0]}) for g in gridcells_list
                ]
        return self._grids

    @property
    def df_adhoc(self):
        if self._df_adhoc is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_AdHocObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                f"AND ObservationDate <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_adhoc = self._get_df(query)
        return self._df_adhoc

    @property
    def df_cameratrap_dep(self):
        if self._df_ct_dep is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
                f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_ct_dep = self._get_df(query, "CameraTrapDeploymentID")
        return self._df_ct_dep

    # TODO: modify DB query to only select unique observations for each CameraTrapDeploymentID AND ObservationDateTime (Kim 1)
    @property
    def df_cameratrap_obs(self):
        if self._df_ct_obs is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDateTime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND ObservationDateTime <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_ct_obs = self._get_df(query, "CameraTrapDeploymentID")
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
                # TODO: make sure zero observation deployments are included (Jamie 1)
                self._df_ct = pd.merge(
                    left=_df_ct_dep, right=_df_ct_obs, left_index=True, right_index=True
                )
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

    def tri(self, dem, scale):
        neighbors = dem.neighborhoodToBands(ee.Kernel.square(1.5))
        diff = dem.subtract(neighbors)
        sq = diff.multiply(diff)
        tri = sq.reduce("sum").sqrt().reproject(self.crs, None, scale)
        return tri

    # Currently this just gets the mean of each covariate within each grid cell (based on self.scale = 1km)
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
                        # TODO: need to change the logic for choosing which columns to modify. what is unnamed? 
                        # (Jamie, Kim 3)
                        for col in covar_stats.columns:
                             if not col.startswith("Unnamed"):
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
            self.neg_log_likelihood_int,
            param_guess,
            method="BFGS",
            options={"gtol": 1},
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
        # TODO: This should get initialized in __init__ so that predict_surface won't fail (Jamie 1)
        self.psi = 1.0 - np.exp(-lambda0)
        nll_po = 0

        zeta = np.empty((len(self.psi), 2))
        zeta[:, 0] = 1.0 - self.psi
        # TODO: handle RuntimeWarning divide by zero with log (Jamie 1)
        zeta[:, 1] = np.log(self.psi)
        # TODO: same __init__ issue (Jamie 1)
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
                        self.df_cameratrap[self.df_cameratrap[uniqueid_y] == i][
                            self.MASTER_CELL_LABEL  
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
                self.MASTER_CELL_LABEL  
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
        prob_images = []

        for gridname in self.grids.keys():
            self._gridname = gridname
            self._reset_df_caches()

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

            df_covars = self.get_covariates(gridname)
            # print(df_covars)
            df_covars.to_csv("covars.csv", encoding="utf-8")
            # df_covars = pd.read_csv(
            #    "covars.csv", encoding="utf-8", index_col=self.cell_label
            # )

            self.po_detection_covars = df_covars[["tri", "distance_to_roads"]]
            self.po_detection_covars.insert(0, "Int", 1)
            self.presence_covars = df_covars[["structural_habitat", "hii"]]
            self.presence_covars.insert(0, "Int", 1)
            self.Nx = self.presence_covars.shape[1]
            if not self.df_adhoc.empty:
                self.Nw = self.po_detection_covars.shape[1]
            else:
                self.Nw = 0

            # TODO: set class properties instead of returning (Jamie 2)
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

        # TODO: add (? or otherwise combine) all probability images, one for each grid (Kim, TBD)
        # self.export_image_ee(combined_images, "hab/probability")

    def check_inputs(self):
        super().check_inputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate", default=datetime.now(timezone.utc).date())
    parser.add_argument("-s", "--species", default="Panthera_tigris")
    parser.add_argument("--scenario", default=SCLTask.CANONICAL)
    options = parser.parse_args()
    sclprobcoeff_task = SCLProbabilityCoefficients(**vars(options))
    sclprobcoeff_task.run()
