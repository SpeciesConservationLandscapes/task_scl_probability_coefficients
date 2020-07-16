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
    ee_rootdir = "projects/SCL/v1"
    # TODO: account for species
    ee_pocdir = "Panthera_tigris/geographies/Sumatra"
    inputs = {
        "obs_adhoc": {"maxage": 1},
        "obs_ss": {"maxage": 1},
        "obs_ct": {"maxage": 1},
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
            "ee_path": "projects/SCL/v1/Panthera_tigris/geographies/Sumatra/hab/structural_habitat",
            "maxage": 1,
        },
    }
    grid_label = "GridName"
    cell_label = "GridCellCode"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_aoi_from_ee(
            "{}/{}/sumatra_poc_aoi".format(self.ee_rootdir, self.species)
        )

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
        self._df_ss = None
        self.Nx = 0
        self.Nw = 0
        self.Npsign = 0
        self.NpCT = 0
        self.po_detection_covars = None  # coefficients relevant to presence-only and background detection only
        self.presence_covars = None  # coefficients relevant to occupancy, shared across models

    def _reset_df_caches(self):
        self._df_adhoc = None
        self._df_ct_dep = None
        self._df_ct_obs = None
        self._df_ss = None

    def _get_df(self, query):
        _gridname_clause = ""
        if self._gridname:
            _gridname_clause = f"AND {self.grid_label} = '{self._gridname}' "
        query = f"{query} {_gridname_clause}"
        df = pd.read_sql(query, self.obsconn)
        df.set_index(self.cell_label, inplace=True)
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
                f"AND ObservationDate <= Cast('{self.taskdate}' AS datetime)"
            )
            self._df_adhoc = self._get_df(query)
        return self._df_adhoc

    # TODO: refactor these CT dfs once we figure out new schema (use adhoc/ss as recipe)
    @property
    def df_cameratrap_dep(self):
        if self._df_ct_dep is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
                f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime)"
            )
            self._df_ct_dep = pd.read_sql(query, self.obsconn)
        return self._df_ct_dep

    @property
    def df_cameratrap_obs(self):
        if self._df_ct_obs is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDateTime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND ObservationDateTime <= Cast('{self.taskdate}' AS datetime)"
            )
            self._df_ct_obs = pd.read_sql(query, self.obsconn)
        return self._df_ct_obs

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_SignSurveyObservation "
                f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                f"AND StartDate <= Cast('{self.taskdate}' AS datetime)"
            )
            self._df_ss = self._get_df(query)
        return self._df_ss

    def fc2df(self, featurecollection):
        features = featurecollection.getInfo()["features"]
        rows = []
        for f in features:
            attr = f["properties"]
            rows.append(attr)

        df = pd.DataFrame(rows)
        df.set_index(self.cell_label, inplace=True)
        return df

    def tri(self, dem, scale):
        neighbors = dem.neighborhoodToBands(ee.Kernel.square(1.5))
        diff = dem.subtract(neighbors)
        sq = diff.multiply(diff)
        tri = sq.reduce("sum").sqrt().reproject(self.crs, None, scale)
        return tri

    # Currently this just gets the mean of each covariate within each grid cell (based on self.scale = 1km)
    # Probably we need more sophisticated covariate definitions (mode of rounded cell vals?)
    # or to sample using smaller gridcell geometries
    def get_covariates(self, grid):
        try:
            cells = [ee.Feature(g[0], g[1]) for g in self.grids[grid]]
        except KeyError:
            raise KeyError(f"No grid {grid} in observations")
        cell_features = ee.FeatureCollection(cells)

        sh_ic = ee.ImageCollection(self.inputs["structural_habitat"]["ee_path"])
        hii_ic = ee.ImageCollection(self.inputs["hii"]["ee_path"])
        dem = ee.Image(self.inputs["dem"]["ee_path"])
        # TODO: when we have OSM, point to fc dir and implement get_most_recent_featurecollection
        roads = ee.FeatureCollection(self.inputs["roads"]["ee_path"])

        structural_habitat, sh_date = self.get_most_recent_image(sh_ic)
        hii, hii_date = self.get_most_recent_image(hii_ic)
        tri = self.tri(dem, 90)
        distance_to_roads = roads.distance().clipToCollection(cell_features)

        if structural_habitat and hii:
            covariates_bands = (
                structural_habitat.rename("structural_habitat")
                .addBands(hii.rename("hii"))
                .addBands(tri.rename("tri"))
                .addBands(distance_to_roads.rename("distance_to_roads"))
            )
            covariates_fc = covariates_bands.reduceRegions(
                collection=cell_features,
                reducer=ee.Reducer.mean(),
                scale=self.scale,
                crs=self.crs,
            )
            _df_covars = self.fc2df(covariates_fc)

            covar_stats = _df_covars.describe()
            # TODO: add comment explaining why we're replacing raw covar values
            # TODO: reexamine this when we figure out more sophisticated sampling -- maybe roll this into ee
            for col in covar_stats.columns:
                if not col.startswith("Unnamed"):
                    _df_covars[col] = (
                            (_df_covars[col] - covar_stats[col]["mean"])
                            / covar_stats[col]["std"]
                    )

            # TODO: check this -- means no row for any cell with ANY missing covars
            return _df_covars.dropna()
        else:
            return None

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

        beta = par[0: self.Nx]
        alpha = par[self.Nx: self.Nx + self.Nw]
        p_sign = expit(par[self.Nx + self.Nw: self.Nx + self.Nw + self.Npsign])
        p_cam = expit(
            par[
                self.Nx
                + self.Nw
                + self.Npsign: self.Nx
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
            zeta[self.df_signsurvey[self.cell_label][j] - 1, 1] = (
                zeta[self.df_signsurvey[self.cell_label][j] - 1, 1]
                + (self.df_signsurvey["detections"][j])
                * np.log(p_sign[self.df_signsurvey["SignSurveyID"][j] - 1])
                + (
                    self.df_signsurvey["NumberOfReplicatesSurveyed"][j]
                    - self.df_signsurvey["detections"][j]
                )
                * np.log(1.0 - p_sign[self.df_signsurvey["SignSurveyID"][j] - 1])
            )

        # TODO: make variable names more readable
        one = self.df_signsurvey[self.df_signsurvey["detections"] > 0][self.cell_label]
        two = CT[CT["det"] > 0][self.cell_label]
        known_occurrences = list(set(one.append(two)))

        zeta[np.array(known_occurrences) - 1, 0] = 0

        lik_so = []
        for i in range(0, len(zeta[:, 0])):
            if zeta[i, 0] == 0:
                lik_so.append(zeta[i, 1])
            else:
                lik_so.append(np.log(zeta[i, 0]) + zeta[i, 1])

        nll_po = -1.0 * (
            + sum(np.log(lambda0[po_data - 1] * p_thin[po_data - 1]))
        )
        nll_so = -1.0 * sum(lik_so)

        return nll_po[0] + nll_so

    # TODO: DRY this up to avoid the repeated lines (should look like neg_log_likelihood_int after refactoring params)
    def predict_surface(self, par, CT, df_signsurvey, griddata):
        """Create predicted probability surface for each grid cell.
         Par: list of parameter values that have been optimized to convert to probability surface
         Returns data frame that includes grid code, grid cell number and predicted probability surface for each grid
         cell"""

        par = np.array(par)
        beta = par[0: self.Nx]
        p_sign = expit(par[self.Nx + self.Nw: self.Nx + self.Nw + self.Npsign])
        p_cam = expit(
            par[
                self.Nx
                + self.Nw
                + self.Npsign: self.Nx
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
        cond_psi = [zeta[i, 1] / sum(zeta[i,:]) for i in range(0, len(psi))]

        cond_prob = 1.0 - (1.0 - np.exp(np.multiply(-1, cond_psi)))
        gridcells = [i for i in range(1, len(psi) + 1)]
        temp = {
            self.cell_label: griddata[self.cell_label],
            "gridcells": gridcells,
            "condprob": cond_prob,
        }
        prob_out = pd.DataFrame(
            temp, columns=[self.cell_label, "gridcells", "condprob"]
        )

        return prob_out

    def calc(self):
        prob_images = []
        for gridname in self.grids.keys():
            self._gridname = gridname
            self._reset_df_caches()
            # just observations for this gridname, where cell labels can be used as index
            # print(self.df_adhoc)
            # print(self.df_signsurvey)
            # TODO: combine CT dep and obs dfs for prob functions
            # df_covars = self.get_covariates(gridname)
            # print(df_covars)
            # df_covars.to_csv("covars.csv", encoding="utf-8")
            # df_covars = pd.read_csv(
            #     "covars.csv", encoding="utf-8", index_col=self.cell_label
            # )

            # TODO: set these dynamically
            self.Nx = 3
            self.Nw = 3
            self.Npsign = 1
            self.NpCT = 1

            # self.po_detection_covars = df_covars[["tri", "distance_to_roads"]]
            # # TODO: Why do we need the extra columns? Can 'alpha' and 'beta' be added to these dfs here?
            # self.po_detection_covars.insert(0, "Int", 1)
            # self.presence_covars = df_covars[["structural_habitat", "hii"]]
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
            # self.export_image_ee(fake_prob, f"{self.ee_pocdir}/hab/probability")

        # TODO: add (? or otherwise combine) all probability images, one for each grid
        # self.export_image_ee(combined_images, f"{self.ee_pocdir}/hab/probability")

    def check_inputs(self):
        super().check_inputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate", default=datetime.now(timezone.utc).date())
    parser.add_argument("-s", "--species", default="Panthera_tigris")
    options = parser.parse_args()
    sclprobcoeff_task = SCLProbabilityCoefficients(**vars(options))
    sclprobcoeff_task.run()
