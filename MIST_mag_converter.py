#%% MODULES AND FUNCTIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits


def app_to_abs(app_mag, distance_parsec, extinction=0):
    return app_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


def abs_to_app(abs_mag, distance_parsec, extinction=0):
    return abs_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


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

#%% LIMITING TO <1.4SOL_MASS
photometry = photometry[features["star_mass"] <= 1.4]
features = features[features["star_mass"] <= 1.4]
# features.describe()

#%% MASS LIST
mass_difference_threshold = 0.006
mass_list = []
initial_mass = 0
for mass in features["star_mass"]:
    if abs(initial_mass - mass) >= mass_difference_threshold:
        mass_list.append(mass)
    initial_mass = mass

#%% BOOLEAN MASK AND PHOTOMETRY
age = 790 * 10 ** 6

bool_list = [
    [
        True if abs(star_mass - mass_ele) <= mass_difference_threshold else False
        for star_mass in features["star_mass"]
    ]
    for mass_ele in mass_list
]

photometry_df = pd.concat(
    [
        photometry.iloc[
            features[booleans]
            .index[(features[booleans]["star_age"] - age).abs().argsort()[:2]]
            .to_list()
        ]
        for booleans in bool_list
    ]
)

#%% CLUSTER DATA IMPORTED
# file_path = "D:/dev/spin_down/new_data/m50/irwin_2009.tsv"
# cluster = pd.read_csv(file_path, comment="#", delimiter="\t", skipinitialspace=True)

file_path = "D:/dev/spin_down/new_data/praesepe/rebull_2017.tsv"
cluster = pd.read_csv(file_path, comment="#", delimiter="\t", skipinitialspace=True)
cluster.describe()
#%% .
distance = 184
extinction = 0.02
# star_mag = app_to_abs(15.911, distance, extinction)

# chosen_mass_index = photometry_df.iloc[
#     (photometry_df["2MASS_Ks"] - star_mag).abs().argsort()[:1]
# ].index[0]
# features.iloc[chosen_mass_index].star_mass

cluster_mass = [
    features.iloc[
        photometry_df.iloc[
            (photometry_df["2MASS_Ks"] - app_to_abs(mag, distance, extinction))
            .abs()
            .argsort()[:1]
        ].index[0]
    ].star_mass
    for mag in cluster.Ksmag.to_numpy()
]

fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.scatter(
    photometry_df["2MASS_Ks"],
    features.star_mass[photometry_df["2MASS_Ks"].index.to_list()],
)

ax.scatter(app_to_abs(cluster.Ksmag.to_numpy(), distance, extinction), cluster_mass)


fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.invert_xaxis()
ax.scatter(cluster_mass, cluster.Per.to_numpy())
#%%
