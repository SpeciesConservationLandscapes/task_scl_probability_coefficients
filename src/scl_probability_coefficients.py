import os
import argparse
import pyodbc
import pandas as pd
import ee
from datetime import datetime, timezone
from task_base import SCLTask
from geomet import wkt


def fc2df(featurecollection):
    features = featurecollection.getInfo()["features"]
    rows = []
    for f in features:
        attr = f["properties"]
        rows.append(attr)

    return pd.DataFrame(rows)


class SCLProbabilityCoefficients(SCLTask):
    ee_rootdir = "projects/SCL/v1"
    inputs = {
        "obs_adhoc": {"maxage": 10},
        "obs_ss": {"maxage": 10},
        "obs_ct": {"maxage": 10},
        "hii": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "projects/HII/v1/hii",
            "maxage": 1,
        },
        "dem": {
            "ee_type": SCLTask.IMAGE,
            "ee_path": "CGIAR/SRTM90_V4",
        },  # no maxage; SRTM 20 years old
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_aoi_from_ee(
            "{}/{}/sumatra_poc_aoi".format(self.ee_rootdir, self.species)
        )

        self._df_adhoc = None
        self._df_ct_dep = None
        self._df_ct_obs = None
        self._df_ss = None
        self._grids = {}

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

    # could skip dataframes and go directly to numpy arrays, or use
    # https://pandas.pydata.org/pandas-docs/version/0.24.0rc1/api/generated/pandas.Series.to_numpy.html
    # depends on calculation needs
    # TODO: account for species
    @property
    def df_adhoc(self):
        if self._df_adhoc is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_AdHocObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDate, {self.taskdate}) <= {self.inputs['obs_adhoc']['maxage']}"
            )
            self._df_adhoc = pd.read_sql(query, self.obsconn)
        return self._df_adhoc

    @property
    def df_cameratrap_dep(self):
        if self._df_ct_dep is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
                f"WHERE DATEDIFF(YEAR, PickupDatetime, {self.taskdate}) <= {self.inputs['obs_ct']['maxage']}"
            )
            self._df_ct_dep = pd.read_sql(query, self.obsconn)
        return self._df_ct_dep

    @property
    def df_cameratrap_obs(self):
        if self._df_ct_obs is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDateTime, {self.taskdate}) <= {self.inputs['obs_ct']['maxage']}"
            )
            self._df_ct_obs = pd.read_sql(query, self.obsconn)
        return self._df_ct_obs

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            query = (
                f"SELECT * FROM dbo.vw_CI_SignSurveyObservation "
                f"WHERE DATEDIFF(YEAR, StartDate, {self.taskdate}) <= {self.inputs['obs_ss']['maxage']}"
            )
            self._df_ss = pd.read_sql(query, self.obsconn)
        return self._df_ss

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
                    ee.Feature(wkt.loads(g[1]), {"gridcellcode": g[0]})
                    for g in gridcells_list
                ]
        return self._grids

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
            cells = self.grids[grid]
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
            return fc2df(covariates_fc)
        else:
            return None

    def calc(self):
        print(self.df_adhoc)
        print(self.df_cameratrap_dep)
        print(self.df_cameratrap_obs)
        print(self.df_signsurvey)
        for gridname in self.grids.keys():
            df_covars = self.get_covariates(gridname)
            print(df_covars)

        # TODO: calculate coefficients here (port R code)

        # store results in ee table in form {"coeff": "<coeff name>", "value": <float | 0 | None>}
        # dummy_geom = ee.Geometry.Point([0, 0])  # can't export tables with undefined or null geometries
        # structural_habitat = ee.Feature(dummy_geom, {"coeff": "structural_habitat", "value": 9.543099})
        # tri = ee.Feature(dummy_geom, {"coeff": "tri", "value": None})
        # hii = ee.Feature(dummy_geom, {"coeff": "hii", "value": 13.821215})
        # distance_to_roads = ee.Feature(dummy_geom, {"coeff": "distance_to_roads", "value": 98.64668})
        # fc = ee.FeatureCollection([structural_habitat, tri, hii, distance_to_roads])
        # self.export_fc_ee(fc, f"{self.species}/hab/probability_coefficients")

    def check_inputs(self):
        super().check_inputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate", default=datetime.now(timezone.utc).date())
    parser.add_argument("-s", "--species", default="Panthera_tigris")
    options = parser.parse_args()
    sclprobcoeff_task = SCLProbabilityCoefficients(**vars(options))
    sclprobcoeff_task.run()
