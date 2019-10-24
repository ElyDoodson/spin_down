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


def make_mass_list(mass_data, threshold=0.006):
    """
    Takes a list of masses that increase step-like and returns the value of the steps.
    E.g. [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.25, 0.25, 0.3] -> [0.1, 0.2, 0.25, 0.3]

    parameters
    --------------
        mass_list:  array-like
            b
    """
    mass_list = []
    initial_mass = 0
    for mass in mass_data:
        if abs(initial_mass - mass) >= threshold:
            mass_list.append(mass)
        initial_mass = mass
    return mass_list


def make_bool_list(mass_step, mass_dataframe, threshold=0.006):
    return [
        True if abs(star_mass - mass_step) <= threshold else False
        for star_mass in mass_dataframe
    ]


def make_bool_mask(mass_dataframe, threshold=0.006):
    return [
        make_bool_list(mass_item, mass_dataframe, threshold)
        for mass_item in make_mass_list(mass_dataframe, threshold)
    ]


def segment_dataframe(data_frame, mask):
    """
    Separates a given DataFrame into a list given by a boolean mask.
    e.g. df.index =  1, 2, 3  mask =[T,F,F],[F,T,F],[F,F,T] -> [[df.1], [df.2], [df.3]]
    """
    return [data_frame[boolean] for boolean in mask]


def return_age_range(data_frame_list, cluster_age, age_range):
    cluster_age = cluster_age * 10 ** 6
    age_range = age_range * 10 ** 6
    return [
        data_frame[
            (data_frame["star_age"] <= cluster_age + age_range)
            & (data_frame["star_age"] >= cluster_age - age_range)
        ]
        for data_frame in data_frame_list
    ]


def return_magnitude_df(photometry_df_lst, mag_name, mag_value):
    lst = []
    for df in photometry_df_lst:
        if len(df) >= 2:
            lst.append(
                pd.concat(
                    [
                        df[
                            df[mag_name] == df[df[mag_name] > mag_value][mag_name].min()
                        ],
                        df[
                            df[mag_name] == df[df[mag_name] < mag_value][mag_name].max()
                        ],
                    ]
                )
            )
        elif len(df) == 1:
            lst.append(df)
        elif len(df) == 0:
            lst.append(None)
    return lst


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

#%% MASS LIST
# mass_difference_threshold = 0.006
# mass_list = []
# initial_mass = 0
# for mass in features["star_mass"]:
#     if abs(initial_mass - mass) >= mass_difference_threshold:
#         mass_list.append(mass)
#     initial_mass = mass

#%% BOOLEAN MASK AND PHOTOMETRY
# age = 790 * 10 ** 6

# bool_list = [
#     [
#         True if abs(star_mass - mass_ele) <= mass_difference_threshold else False
#         for star_mass in features["star_mass"]
#     ]
#     for mass_ele in mass_list
# ]

# photometry_df = pd.concat(
#     [
#         photometry.iloc[
#             features[booleans]
#             .index[(features[booleans]["star_age"] - age).abs().argsort()[:2]]
#             .to_list()
#         ]
#         for booleans in bool_list
#     ]
# )
#%% CLUSTER DATA IMPORTED
# file_path = "D:/dev/spin_down/new_data/m50/irwin_2009.tsv"
# cluster = pd.read_csv(file_path, comment="#", delimiter="\t", skipinitialspace=True)

file_path = "D:/dev/spin_down/new_data/praesepe/rebull_2017.tsv"
cluster = pd.read_csv(file_path, comment="#", delimiter="\t", skipinitialspace=True)
cluster.describe()
#%% KNN = 1
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
ax.invert_xaxis()
ax.scatter(
    cluster_mass,
    app_to_abs(cluster.Ksmag.to_numpy(), distance, extinction),
    c="green",
    label="Cluster Data",
)
ax.scatter(
    features.star_mass[photometry_df["2MASS_Ks"].index.to_list()],
    photometry_df["2MASS_Ks"],
    marker="x",
    c="white",
    s=10,
    label="MIST data",
)
ax.legend()
ax.set(xlabel="Mass (M_Solar)", ylabel="Abs K Magnitude (mag)")


fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.invert_xaxis()
ax.set(title="Praesepe", xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(cluster_mass, cluster.Per.to_numpy(), c="green")

#%%
# segmented_df = segment_dataframe(features, make_bool_mask(features["star_mass"]))

# df = photometry.iloc[return_age_range(segmented_df, 790, 100)[2].index]
# # df[df["2MASS_Ks"] > 6.027]["2MASS_Ks"].min()
# # df[df["2MASS_Ks"] == df[df["2MASS_Ks"] < 6.027]["2MASS_Ks"].max()]

# # features.iloc[pd.concat(return_magnitude_df(
# #     [photometry.iloc[item.index] for item in return_age_range(segmented_df, 790, 50)],
# #     "2MASS_Ks",
# #     6.027,
# # )).index]
# fig, ax = plt.subplots(1, figsize=(11.5, 7))
# ax.set(xlabel="Stellar Age (Yrs)", ylabel="Kmag")
# ax.plot(
#     segmented_df[0]
#     .loc[
#         (segmented_df[0].star_age >= 6 * 10 ** 6)
#         & (segmented_df[0].star_age <= 10 * 10 ** 8)
#     ]
#     .star_age,
#     photometry.iloc[
#         segmented_df[0]
#         .loc[
#             (segmented_df[0].star_age >= 6 * 10 ** 6)
#             & (segmented_df[0].star_age <= 10 * 10 ** 8)
#         ]
#         .index
#     ]["2MASS_Ks"],
#     label="Mass = 0.10 M_Solar",
# )

# ax.plot(
#     segmented_df[1]
#     .loc[
#         (segmented_df[1].star_age >= 6 * 10 ** 6)
#         & (segmented_df[1].star_age <= 10 * 10 ** 8)
#     ]
#     .star_age,
#     photometry.iloc[
#         segmented_df[1]
#         .loc[
#             (segmented_df[1].star_age >= 6 * 10 ** 6)
#             & (segmented_df[1].star_age <= 10 * 10 ** 8)
#         ]
#         .index
#     ]["2MASS_Ks"],
#     label="Mass = 0.15 M_Solar",
# )

# fake_kmag = np.linspace(6.5, 9.5, 100)

# ax.plot([6e8] * len(fake_kmag), fake_kmag, linestyle="--", label="Lower bound")
# ax.plot([10e8] * len(fake_kmag), fake_kmag, linestyle="--", label="Upper bound")
# ax.legend()

#%%
age = 790 * 10 ** 6
age_err = 150 * 10 ** 6

df = photometry.iloc[
    features[
        (features.star_age >= age - age_err) & (features.star_age <= age + age_err)
    ].index
]


display(one_above_below(df, "2MASS_Ks", 9))
display(features.iloc[one_above_below(df, "2MASS_Ks", 9).index])

x = one_above_below(df, "2MASS_Ks", 9)["2MASS_Ks"].to_numpy()
index = x.argsort()
x = x[index]
y = features.iloc[one_above_below(df, "2MASS_Ks", 9).index].star_mass.to_numpy()[index]

np.interp(9, x, y)
#%%
mass = []
for kmag in cluster.Ksmag.to_numpy():
    # kmag = star.Ksmag
    abs_kmag = app_to_abs(kmag, 184, 0.027)

    x = one_above_below(df, "2MASS_Ks", abs_kmag)["2MASS_Ks"].to_numpy()
    index = x.argsort()
    x = x[index]
    y = features.iloc[
        one_above_below(df, "2MASS_Ks", abs_kmag).index
    ].star_mass.to_numpy()[index]
    mass.append(np.interp(abs_kmag, x, y))

fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.invert_xaxis()
ax.set(title="Praesepe", xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, cluster.Per.to_numpy(), color="green")

#%%
dictionary = {
    "hyades": {"age": 50, "data": [1, 2]},
    "h_per": {"age": 150, "data": [3, 4]},
}

dictionary.update({"hyades": {"age": 51, "data": [1, 2]}})
dictionary
