#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits

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
# list(df2.columns.values)

#%%
photometry = photometry[features["star_mass"] <= 1.4]
features = features[features["star_mass"] <= 1.4]
# features.describe()

#%%
age = 500 * 10 ** 6
# features.iloc[(features['star_age']/10**6 - age).abs().argsort()[:2]]
# features.loc[abs(features['star_age'] - age) <= 10000000]


fig, ax = plt.subplots(1, figsize=(11.5, 7))

ax.plot(features["star_mass"])


#%%
mass_difference_threshold = 0.006
mass_list = []
initial_mass = 0
for mass in features["star_mass"]:
    if abs(initial_mass - mass) >= mass_difference_threshold:
        mass_list.append(mass)
    initial_mass = mass

#%%
bool_list = [
    [
        True if abs(star_mass - mass_ele) <= mass_difference_threshold else False
        for star_mass in features["star_mass"]
    ]
    for mass_ele in mass_list
]

photometry_list = [
    photometry.iloc[
        features[booleans]
        .index[(features["star_age"] / 10 ** 6 - age).abs().argsort()[:2]]
        .to_list()
    ]
    for booleans in bool_list
]

kmag_list = np.array([ pair["2MASS_Ks"].to_list() for pair in photometry_list])

star_mag = 5.6
min(np.concatenate(kmag_list), key=lambda x:abs(x-star_mag))
