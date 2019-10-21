#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits


def app_to_abs(app_mag, distance_parsec):
    return app_mag - (5 * np.log10(distance_parsec)) + 5


def abs_to_app(abs_mag, distance_parsec):
    return abs_mag - (5 * np.log10(distance_parsec)) + 5


#%%

filename1 = "new_MIST_data/MIST_V1.2_feh0_afe0.fits"
data1 = Table.read(filename1, format="fits")
features = data1.to_pandas()
# features.describe()
# list(features.columns.values)

#%%
filename2 = "new_MIST_data/MIST_V1.2_feh0_afe0_wPHOT.fits"
data2 = Table.read(filename2, format="fits")
photometry = data2.to_pandas()
# photometry.describe()
# list(photometry.columns.values)

#%%
photometry = photometry[features["star_mass"] <= 1.4]
features = features[features["star_mass"] <= 1.4]
# features.describe()

#%%
# age = 78 * 10 ** 6
# # features.iloc[(features['star_age']/10**6 - age).abs().argsort()[:2]]
# # features.loc[abs(features['star_age'] - age) <= 10000000]


# fig, ax = plt.subplots(1, figsize=(11.5, 7))

# ax.plot(features["star_mass"])


#%%
mass_difference_threshold = 0.006
mass_list = []
initial_mass = 0
for mass in features["star_mass"]:
    if abs(initial_mass - mass) >= mass_difference_threshold:
        mass_list.append(mass)
    initial_mass = mass

#%%
age = 130 * 10 ** 6

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
            .index[(features[booleans]["star_age"]  - age).abs().argsort()[:2]]
            .to_list()
        ]
        for booleans in bool_list
    ]
)

# kmag_list = np.array([pair["2MASS_Ks"].to_list() for pair in photometry_list])

# star_mag = 5.6
# min(np.concatenate(kmag_list), key=lambda x: abs(x - star_mag))

#%%
file_path = "D:/dev/spin_down/new_data/m50/irwin_2009.tsv"
cluster = pd.read_csv(file_path, comment="#", delimiter="\t", skipinitialspace=True)

#%%
distance = 881
star_mag = app_to_abs(15.911, distance)

chosen_mass_index = photometry_df.iloc[
    (photometry_df["Bessell_I"] - star_mag).abs().argsort()[:1]
].index[0]

# features.iloc[chosen_mass_index].star_mass
fig, ax = plt.subplots(1, figsize=(11.5, 7))

photometry_df.describe()
plt.scatter(
    photometry_df["Bessell_I"],
    features.star_mass[photometry_df["Bessell_I"].index.to_list()],
)
plt.scatter(app_to_abs(cluster.Icmag.to_numpy(),distance), cluster.Mass.to_list())

#%%


#%%
