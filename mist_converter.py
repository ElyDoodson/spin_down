#%% MODULEs BEING INSTANTIATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits

#%% FUNCTIONS BEING DECLARED
def app_to_abs(app_mag, distance_parsec, extinction=0):
    return app_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


def abs_to_app(abs_mag, distance_parsec, extinction=0):
    return abs_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


def one_above_below(df, mag_name, mag_value):
    return pd.concat(
        [
            df[df[mag_name] == df[df[mag_name] > mag_value][mag_name].min()],
            df[df[mag_name] == df[df[mag_name] < mag_value][mag_name].max()],
        ]
    )


#%% READING AND ASSIGNING FIT FILES
filename1 = "new_MIST_data/MIST_V1.2_feh0_afe0.fits"
data1 = Table.read(filename1, format="fits")
features = data1.to_pandas()
# features.describe()
# list(features.columns.values)

filename2 = "new_MIST_data/MIST_V1.2_feh0_afe0_wPHOT.fits"
data2 = Table.read(filename2, format="fits")
photometry = data2.to_pandas()
# photometry.describe()
# list(photometry.columns.values)

#%% LIMITING TO <1.7SOL_MASS
photometry = photometry[features["star_mass"] <= 1.7]
features = features[features["star_mass"] <= 1.7]
# features.describe()
