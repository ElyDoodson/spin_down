#%% MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

#%% FUNCTIONS


def app_to_abs(app_mag, distance_parsec):
    return app_mag - (5 * np.log10(distance_parsec)) + 5


#%%

path = "D:/dev/spin_down/new_data/"

cluster_list = os.listdir(path)
files = [os.listdir(path + str(cluster)) for cluster in cluster_list]

dict_list = [
    {
        "name": cluster_list[i],
        "data_frames": [
            pd.read_csv(
                filepath_or_buffer=path + cluster_list[i] + "/" + file,
                comment="#",
                delimiter="\t",
                skipinitialspace=True,
            )
            for file in file_list
        ],
    }
    for i, file_list in enumerate(files)
]

for index, item in enumerate(dict_list):
    item.update({"author": files[index]})


#%%
color_chart = pd.read_csv("D:/dev/spin_down/colour_chart.csv", comment="#", sep=" ")

fit_vk_bv = LinearRegression()
fit_vk_bv.fit(
    [[value] for value in color_chart.V_KS.to_numpy()], color_chart.B_V.to_numpy()
)

# VALUES FROM Wright et al. 2011
wright_v_k = [
    [1.14, 1.48],
    [1.48, 1.79],
    [1.81, 2.22],
    [2.23, 2.80],
    [2.81, 3.34],
    [3.36, 3.68],
    [3.69, 4.19],
    [4.21, 4.62],
    [4.63, 4.93],
    [4.95, 6.61],
]
wright_b_v = [
    [0.46, 0.61],
    [0.61, 0.75],
    [0.76, 0.92],
    [0.92, 1.12],
    [1.13, 1.31],
    [1.32, 1.41],
    [1.41, 1.49],
    [1.50, 1.55],
    [1.55, 1.60],
    [1.61, 1.95],
]
wright_mass = [
    [1.16, 1.36],
    [1.02, 1.16],
    [0.89, 1.02],
    [0.77, 0.89],
    [0.63, 0.77],
    [0.47, 0.62],
    [0.26, 0.47],
    [0.18, 0.25],
    [0.14, 0.18],
    [0.09, 0.14],
]


fit_vk_mass = LinearRegression()
fit_vk_mass.fit([[np.average(item)] for item in wright_v_k], wright_mass)

fit_bv_mass = LinearRegression()
fit_bv_mass.fit([[np.average(item)] for item in wright_b_v], wright_mass)
#%% ALPHA PER
pm_list = [[] for i in range(len(dict_list))]


pm_list[0] = {
    "Per": [item for item in dict_list[0]["data_frames"][0].Prot.to_numpy()],
    "Mass": [
        item[1]
        for item in fit_bv_mass.predict(
            [[item] for item in dict_list[0]["data_frames"][0]["B-V"].to_numpy()]
        )
    ],
}

pm_list[0] = {
    "Per": dict_list[0]["data_frames"][0].Prot.to_numpy(),
    "Mass": [
        item[1]
        for item in fit_bv_mass.predict(
            dict_list[0]["data_frames"][0]["B-V"].to_numpy().reshape(-1, 1)
        )
    ],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[0]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[0]["Mass"], pm_list[0]["Per"])


#%% HYADES
vmag = dict_list[1]["data_frames"][0].Vmag.to_numpy()
kmag = dict_list[1]["data_frames"][0].Ksmag.to_numpy()
v_k = vmag - kmag
period_0 = dict_list[1]["data_frames"][0].Per.to_numpy()[
    ~np.isnan(dict_list[1]["data_frames"][0].Per.to_numpy())
][~np.isnan(v_k)]
v_k = v_k[~np.isnan(v_k)]


pm_list[1] = {
    "Per": np.append(
        period_0, dict_list[1]["data_frames"][1].IPer.to_numpy()
    ),  # dict_list[1]["data_frames"][0].Per.to_numpy(),
    "Mass": np.append(
        [item[1] for item in fit_vk_mass.predict(v_k.reshape(-1, 1))],
        dict_list[1]["data_frames"][1].Mass.to_numpy(),
    ),
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[1]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[1]["Mass"], pm_list[1]["Per"])

#%% M34

b_v = dict_list[2]["data_frames"][0]["(B-V)0"].to_numpy()

period_0 = dict_list[2]["data_frames"][0].Prot.to_numpy()[~np.isnan(b_v)]
b_v = b_v[~np.isnan(b_v)]
pm_list[2] = {
    "Per": period_0,
    "Mass": [item[1] for item in fit_bv_mass.predict(b_v.reshape(-1, 1))],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[2]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[2]["Mass"], pm_list[2]["Per"])

#%% M35
# dict_list[3]["data_frames"][0]
b_v = dict_list[3]["data_frames"][0]["(B-V)0"].to_numpy()

period_0 = dict_list[3]["data_frames"][0].Prot.to_numpy()[~np.isnan(b_v)]
b_v = b_v[~np.isnan(b_v)]

pm_list[3] = {
    "Per": period_0,
    "Mass": [item[1] for item in fit_bv_mass.predict(b_v.reshape(-1, 1))],
}
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[3]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[3]["Mass"], pm_list[3]["Per"])

#%% M37
# dict_list[4]["data_frames"][0].describe()
pm_list[4] = {
    "Per": dict_list[4]["data_frames"][0].Per.to_numpy(),
    "Mass": [
        item[1]
        for item in fit_bv_mass.predict(
            np.subtract(
                dict_list[4]["data_frames"][0].Bmag.to_numpy(),
                dict_list[4]["data_frames"][0].Vmag.to_numpy() + 0.2,
            ).reshape(-1, 1)
        )
    ],
}
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[4]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[4]["Mass"], pm_list[4]["Per"])

#%% NGC6811
dict_list[5]["data_frames"][0].describe()

pm_list[5] = {
    "Per": dict_list[5]["data_frames"][0].Per.to_numpy(),
    "Mass": [
        item[1]
        for item in fit_bv_mass.predict(
            1.02
            * np.subtract(
                dict_list[5]["data_frames"][0].gmag.to_numpy(),
                dict_list[5]["data_frames"][0].rmag.to_numpy(),
            ).reshape(-1, 1)
            + 0.2
            - 0.1
        )
    ],
}

# len(pm_list[5]["Per"])
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_list[5]["name"], xlim=(1.8, 0.0))
ax.scatter(pm_list[5]["Mass"], pm_list[5]["Per"])


#%% PLEIADES
print(dict_list[6]["data_frames"][1].describe())

pm_list[6] = {
    "Per": dict_list[6]["data_frames"][1].Prot.to_numpy(),
    "Mass":[item[1] for item in fit_vk_mass.predict(dict_list[6]["data_frames"][1]["(V-K)0"].to_numpy().reshape(-1,1))],
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_list[6]["name"])#, xlim=(1.8, 0.0))
ax.scatter(pm_list[6]["Mass"], pm_list[6]["Per"])

#%% PRAESEPE
print(dict_list[7]["data_frames"][2].describe())

period_2 = dict_list[7]["data_frames"][2].Per.to_numpy()

v_k = dict_list[7]["data_frames"][2]["V-Ks"].to_numpy()[~np.isnan(period_2)]
period_2 = period_2[~np.isnan(period_2)]

pm_list[7] = {
    "Per": period_2,
    "Mass": [item[1] for item in fit_vk_mass.predict(v_k.reshape(-1,1))]
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_list[7]["name"])#, xlim=(1.8, 0.0))
ax.scatter(pm_list[7]["Mass"], pm_list[7]["Per"])

#%% UPPER SCORPION
print(len(dict_list[8]["data_frames"][0].Per))

period_0 = dict_list[8]["data_frames"][0].Per.to_numpy()

v_k = dict_list[8]["data_frames"][0]["(V-Ks)0"].to_numpy()[~np.isnan(period_0)]

period_0 = period_0[~np.isnan(period_0)]

period_0 = period_0[~np.isnan(v_k)]
v_k = v_k[~np.isnan(v_k)]

print(len(v_k))
pm_list[8] = {
    "Per": period_0,
    "Mass": [item[1] for item in fit_vk_mass.predict(v_k.reshape(-1,1))]
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_list[8]["name"])#, xlim=(1.8, 0.0))
ax.scatter(pm_list[8]["Mass"], pm_list[8]["Per"])
