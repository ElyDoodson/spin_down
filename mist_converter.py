#%% MODULEs BEING INSTANTIATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits
import matplotlib.gridspec as gridspec

plt.style.use("dark_background")
#%% FUNCTIONS BEING DECLARED
def app_to_abs(app_mag, distance_parsec, extinction=0):
    return app_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


def abs_to_app(abs_mag, distance_parsec, extinction=0):
    return abs_mag - (5 * np.log10(distance_parsec)) + 5 - extinction


def one_above_below(df, value_name, value):
    """
    Returns a dataframe with value above and below the specified value 
    """
    return pd.concat(
        [
            df[df[value_name] == df[df[value_name] > value][value_name].min()],
            df[df[value_name] == df[df[value_name] < value][value_name].max()],
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


def mist_tau_interpolate(data_frame, mass_list, age, age_err = 0):
    lst = []
    age_reduced_df = data_frame.loc[
        (data_frame.star_age >= age - age_err) & (data_frame.star_age <= age + age_err)
    ]
    values_above = data_frame[data_frame.star_age > age].sort_values("star_age")[:100]
    values_below = data_frame[data_frame.star_age < age].sort_values(
        "star_age", ascending=False
    )[:100]

    age_reduced_df = data_frame.iloc[pd.concat((values_above, values_below)).index]

    for mass in mass_list:
        # Creates DF from age_reduced df, contraining the mass value 1 above and below chosen value
        oneup_onedown = one_above_below(age_reduced_df, "star_mass", mass)
        x = oneup_onedown["star_mass"].to_numpy()

        # Returns the index of the sorted "above and below" values
        # (this is required for np.interp to work)
        index = x.argsort()

        # The two are sorted ascending
        x = x[index]

        # The coresponding tau values for the two masses is found and used in the linear interpolation
        y = data_frame.iloc[oneup_onedown.index].conv_env_turnover_time_g.to_numpy()[
            index
        ]

        # print(x,y)
        lst.append(np.interp(mass, x, y))
    return np.array(lst)


#%% READING AND ASSIGNING FIT FILES
filename1 = "new_MIST_data/MIST_V1.2_feh0_afe0.fits"
data1 = Table.read(filename1, format="fits")
features = data1.to_pandas()
features_with_tau = pd.read_csv("D:/dev/spin_down/new_MIST_data/tau_mist.csv", sep="\t")
# features_with_tau = features_with_tau.sort_values(by = "star_mass")
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
cluster_dict = {}

#%% Alpha Per
name = "alpha_per"
age = 70 * 10 ** 6
age_err = 50 * 10 ** 6
dist = 165
reddening = 0.11  # B-v
path = "D:/dev/spin_down/new_data/" + "alpha_per/prosser_1997.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Vmag.to_numpy(), dist, reddening)

period = data.Prot.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
#%% Hyades
name = "hyades"
age = 600 * 10 ** 6
age_err = 150 * 10 ** 6
dist = 47
reddening = 0.0
path = "D:/dev/spin_down/new_data/hyades/douglas_2016.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
data = data[~np.isnan(data.K2Per)]
data = data[~np.isnan(data.Mass)]

# data = data[~np.isnan(data.)]

# mag_str = "2MASS_Ks"


# abs_mags = app_to_abs(data.Ksmag.to_numpy(), dist, reddening)

# mass = mist_mass_interpolate(df, features, abs_mags, mag_str)

mass = data.Mass.to_numpy()
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, data.K2Per.to_numpy(), color="green")

data_dict = {"Per": data.K2Per.to_numpy(), "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% H Persei
name = "h_per"
age = 13e6
age_err = 3 * 10 ** 6
dist = 2291  # 11.8 dist mod
reddening = 0
path = "D:/dev/spin_down/new_data/" + "h_per/moraux_2013.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "2MASS_Ks"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Kmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
# mass = data.Mass.to_numpy()
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
#%% M34
name = "m34"
age = 220 * 10 ** 6
age_err = 150 * 10 ** 6
dist = 470
reddening = 0.07  # b-v
path = "D:/dev/spin_down/new_data/" + "m34/meibom_2011.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.V0mag.to_numpy(), dist, reddening)

period = data.Prot.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% M35
name = "m35"
age = 148e6
age_err = 150 * 10 ** 6
dist = 850
reddening = 0
path = "D:/dev/spin_down/new_data/" + "m35/meibom_2009.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]

print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.V0mag.to_numpy(), dist, reddening)

period = data.Prot.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% M37
name = "m37"
age = 550e6
age_err = 15 * 10 ** 6
dist = 1400
reddening = 0.227
path = "D:/dev/spin_down/new_data/" + "m37/chang_2015.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"

# removing nan rows
data = data[~np.isnan(data.Per)]
data = data[~np.isnan(data.Vmag)]


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]

print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Vmag.to_numpy(), dist, reddening)
    

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")


fig, ax = plt.subplots(1, figsize=(8, 4.5))
ax.scatter(tau, period)
ax.set(xlim=(0, 1e7))

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})


#%% M50
name = "m50"
age = 130e6
age_err = 30 * 10 ** 6
dist = 1000
reddening = 0.22  # b-v
path = "D:/dev/spin_down/new_data/" + "m50/irwin_2009.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "SDSS_i"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]

print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Icmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
# mass = data.Mass.to_numpy()
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% NGC2301
name = "ngc2301"
age = 210e6
age_err = 30 * 10 ** 6
dist = 872
reddening = 0.028  # b-v
path = "D:/dev/spin_down/new_data/" + "ngc2301/sukhbold_2009.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_R"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Rmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% NGC2516
name = "ngc2516"
age = 150e6
age_err = 20 * 10 ** 6
dist = 407.4  # 8.05 distance mod
reddening = 0.12  # b-v
path = "D:/dev/spin_down/new_data/" + "ngc2516/irwin_2007.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:200]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:200]

df = photometry.iloc[pd.concat((values_above, values_below)).index]

df2 = photometry.iloc[
    features[
        (features.star_age >= age - age_err) & (features.star_age <= age + age_err)
    ].index
]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Vmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
# mass = data.Mass.to_numpy()
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)

ax[0].invert_xaxis()

ax[0].set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax[0].scatter(mass, period, color="green")

ax[1].hist(features.iloc[df.index].star_mass)


fig, ax = plt.subplots(1, figsize=(8, 4.5))
ax.scatter(tau, period)

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
#%% NGC2547
name = "ngc2547"
age = 38.5e6
age_err = 6 * 10 ** 6
dist = 361.4
reddening = 0.186  # Av
path = "D:/dev/spin_down/new_data/" + "ngc2547/irwin_2008.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Vmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% NGC6811
name = "ngc6811"
age = 1000e6
age_err = 150 * 10 ** 6
dist = 1096
reddening = 0.15  # Av
path = "D:/dev/spin_down/new_data/" + "ngc6811/curtis_2019.csv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "GaiaMAW_G"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Gmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
# mass = data.Mass.to_numpy()
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")


fig, ax = plt.subplots(1, figsize=(8, 4.5))
ax.scatter(tau, period)
ax.set(xlim=(0, 0.2e8))

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% PLEIADES
name = "pleiades"
age = 125e6
age_err = 20 * 10 ** 6
dist = 136
reddening = 0.12  # Av = 0.12, AK = 0.01, E(B-V) = 0.04
path = "D:/dev/spin_down/new_data/" + "pleiades/rebull_2016.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "2MASS_Ks"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Ksmag.to_numpy(), dist, reddening)

period = data.Prot.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% PRAESEPE
name = "praesepe"
age = 790 * 10 ** 6
age_err = 150 * 10 ** 6
dist = 184
reddening = 0.027
path = "D:/dev/spin_down/new_data/praesepe/rebull_2017.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "2MASS_Ks"

# removing nan rows
data = data[~np.isnan(data.Per)]
data = data[~np.isnan(data.Ksmag)]


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Ksmag.to_numpy(), dist, reddening)

mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, data.Per.to_numpy(), color="green")


fig, ax = plt.subplots(1, figsize=(8, 4.5))
ax.scatter(tau, data.Per.to_numpy())
# ax.set(xlim=(0, 0.4e9))


data_dict = {"Per": data.Per.to_numpy(), "Mass": mass}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% UPPER SCORPIO
name = "usco"
age = 8e6
age_err = 3 * 10 ** 6
dist = 170  # estimate
reddening = 0
path = "D:/dev/spin_down/new_data/" + "usco/rebull_2018.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "2MASS_Ks"

# removing nan rows
data = data[~np.isnan(data.Per)]
data = data[~np.isnan(data.Ksmag)]


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.Ksmag.to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})

#%% NGC6819
name = "ngc6819"
age = 2500e6
age_err = 600 * 10 ** 6
dist = 2208
reddening = 0.006  #  0.41 to 0.89 mag
path = "D:/dev/spin_down/new_data/" + "ngc6819/meibom_2015.tsv"


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.V.to_numpy(), dist, reddening)

period = data.PMean.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
tau = mist_tau_interpolate(features_with_tau, mass, age, age_err)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period, "Mass": mass, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})


#%% m67
name = "m67"
age = 4200e6
age_err = 1000 * 10 ** 6
dist = 816
reddening = 0.047  # taylor 2007
path1 = "D:/dev/spin_down/new_data/" + "m67/barnes_2016.csv"
path2 = "D:/dev/spin_down/new_data/" + "m67/gonzalez_2016.tsv"
path3 = "D:/dev/spin_down/new_data/" + "m67/gonzalez2_2016.tsv"


data = pd.read_csv(path1, comment="#", delimiter="\t", skipinitialspace=True)
data2 = pd.read_csv(path2, comment="#", delimiter="\t", skipinitialspace=True)
data3 = pd.read_csv(path3, comment="#", delimiter="\t", skipinitialspace=True)

mag_str = "Bessell_V"


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above, values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data.V.to_numpy(), dist, reddening)
abs_mags2 = app_to_abs(data2.Vmag.to_numpy(), dist, reddening)
abs_mags3 = app_to_abs(data3.Vmag.to_numpy(), dist, reddening)


period = data.P.to_numpy()
period2 = data2.Per.to_numpy()
period3 = data3.Per.to_numpy()

mass = mist_mass_interpolate(df, features, abs_mags, mag_str)
mass2 = mist_mass_interpolate(df, features, abs_mags2, mag_str)
mass3 = mist_mass_interpolate(df, features, abs_mags3, mag_str)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass2, period2, color="pink")

ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass3, period3, color="orange")

period_full = np.concatenate((period, period2, period3))
mass_full = np.concatenate((mass, mass2, mass3))
tau = mist_tau_interpolate(features_with_tau, mass_full, age, age_err)

data_dict = {"Per": period_full, "Mass": mass_full, "Tau": tau}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass", "Tau"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
#%% Plot
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(24, 16))
fig.subplots_adjust(wspace=0.03, hspace=0.05)

cluster_list = np.array(cluster_list)
sorted_list = cluster_list[
    np.argsort([cluster_dict[name]["age"] / 1e6 for name in cluster_list])
]

for num, data in enumerate(sorted_list):
    mass = cluster_dict[data]["data_frame"]["Mass"]
    period = cluster_dict[data]["data_frame"]["Per"]

    r, c = divmod(num, 4)
    # print(r, c)

    ax[r][c].invert_xaxis()

    ax[r][c].plot(
        mass,
        period,
        linestyle="none",
        marker="x",
        label=("%s %iMYrs") % (data, cluster_dict[data]["age"] / 1e6),
        markersize=1.5,
    )

    ax[r][c].legend()
    ax[r][c].set(xlim=(2, 0.0), ylim=(-2, 40.0))  # ylim = (0, 100), yscale = "log",)
#%% Comparing Clsuters
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")

ax.scatter(
    cluster_dict["m37"]["data_frame"].Mass + 0.14,
    cluster_dict["m37"]["data_frame"].Per,
    color="yellow",
    label=cluster_dict["m37"]["age"] / 1e6,
    marker="x",
    s=1.5,
    alpha=0.5,
)
ax.scatter(
    cluster_dict["praesepe"]["data_frame"].Mass,
    cluster_dict["praesepe"]["data_frame"].Per,
    color="green",
    label=cluster_dict["praesepe"]["age"] / 1e6,
    marker="x",
    s=2,
)
ax.scatter(
    cluster_dict["ngc6811"]["data_frame"].Mass,
    cluster_dict["ngc6811"]["data_frame"].Per,
    color="blue",
    label=cluster_dict["ngc6811"]["age"] / 1e6,
    marker="x",
    s=2,
)
ax.scatter(
    cluster_dict["ngc6819"]["data_frame"].Mass,
    cluster_dict["ngc6819"]["data_frame"].Per,
    color="red",
    label=cluster_dict["ngc6819"]["age"] / 1e6,
    marker="x",
    s=2,
)
ax.legend()
#%% Create CSV Files
for name in cluster_list:
    path = "D:/dev/spin_down/mistmade_data/{}_pm_{:e}.csv".format(
        name, cluster_dict[name]["age"] / 1e6
    )
    cluster_dict[name]["data_frame"].to_csv(path_or_buf=path, index=False, sep="\t")


#%% TEST CELL

fig, ax = plt.subplots(1)
ax.set(xlim=(0, 1.2e7))
ax.scatter(cluster_dict["m37"]["data_frame"].Tau, cluster_dict["m37"]["data_frame"].Per)


#%% TEST CELL #2
rossby_ = [
    [
        [cluster_dict[name]["age"], item[1][0], item[1][1]]
        for item in cluster_dict[name]["data_frame"].iterrows()
    ]
    for name in cluster_dict
]

flatten = lambda l: [item for sublist in l for item in sublist]
rossby = pd.DataFrame(flatten(rossby_), columns=["age", "Per", "Tau"])

fig, ax = plt.subplots(1)
# ax.set(xlim = (0,10), ylim = (-0.2,20))
ax.scatter(rossby.Per, rossby.Per / rossby.Tau)


df = rossby
labels, bins = pd.cut(df.Tau, 100, labels=False, retbins=True)

unique_labels = np.unique(labels)

binned_df = [df[labels == label] for label in unique_labels]


fig, ax = plt.subplots(1)
ax.scatter(binned_df[0].age, binned_df[0].Per)

#%% TEMPLATE
"""
name = ..........
age = ..........
age_err = 150*10**6
dist = ............
reddening = ...........
path = "D:/dev/spin_down/new_data/" + ...............


data = pd.read_csv(path, comment="#", delimiter="\t", skipinitialspace=True)
mag_str = .............


values_above = features[features.star_age > age].sort_values("star_age")[:100]
values_below = features[features.star_age < age].sort_values(
    "star_age", ascending=False
)[:100]

df = photometry.iloc[pd.concat((values_above , values_below)).index]
print(
    ("Age Range num: % d, Num in Mass~0.1 % d")
    % (len(df), len(df[features.iloc[df.index].star_mass - 0.1 <= 0.04]))
)
abs_mags = app_to_abs(data. ......... .to_numpy(), dist, reddening)

period = data.Per.to_numpy()
mass = mist_mass_interpolate(df, features, abs_mags, mag_str)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(title=name, xlabel="Mass (M_Solar)", ylabel="Period (days)")
ax.scatter(mass, period, color="green")

data_dict = {"Per": period,  "Mass": mass}
data_frame = pd.DataFrame(data_dict, columns=["Per", "Mass"])
current_dict = {"age": age, "data_frame": data_frame}
cluster_dict.update({name: current_dict})
"""
