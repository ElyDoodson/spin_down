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


def mist_mass_interpolate(photom_dataframe, feature_df, abs_mags, mag_str):
    lst = []
    for abs_mag in abs_mags:
        # Choses a value in photom_dataframe[mag_str] above and below the value of abs_mag
        x = one_above_below(photom_dataframe, mag_str, abs_mag)[mag_str].to_numpy()
        # Returns the index of the sorted "above and below" values
        # (this is required for np.interp to work)
        index = x.argsort()
        # The two are sorted ascending
        x = x[index]
        # The coresponding mass values for the two photometry is found and used in the linear interpolation
        y = feature_df.iloc[
            one_above_below(photom_dataframe, mag_str, abs_mag).index
        ].star_mass.to_numpy()[index]

        lst.append(np.interp(abs_mag, x, y))
    return np.array(lst)


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
photometry = photometry[features["star_mass"] <= 2.0]
features = features[features["star_mass"] <= 2.0]
# features.describe()

#%%
path = "D:/dev/spin_down/new_data/"
cluster_list = os.listdir(path)
files = [os.listdir(path + str(cluster)) for cluster in cluster_list]

#%%
cluster_dict = {}
#%% PRAESEPE
name = "praesepe"
age = 790* 10 ** 6
age_err = 150 * 10 ** 6
dist = 184
reddening = 0.027
path = "D:/dev/spin_down/new_data/praesepe/rebull_2017.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "2MASS_Ks"

df = photometry.iloc[
    features[
        (features.star_age >= age - age_err) & (features.star_age <= age + age_err)
    ].index
]

abs_mags = app_to_abs(data.Ksmag.to_numpy(), dist, reddening)

mass = mist_mass_interpolate(df, features, abs_mags, "2MASS_Ks")

fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, data.Per.to_numpy(), color="green")

data_dict = {"Per": data.Per.to_numpy(), "Mass": mass}
data_frame = pd.DataFrame(data_dict , columns=["Per", "Mass"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
