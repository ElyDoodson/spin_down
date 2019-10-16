#%% MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#%% FUNCTIONS


def app_to_abs(app_mag, distance_parsec):
    return app_mag - (5 * np.log10(distance_parsec)) + 5


def convert_order(lst, order):
    return np.array([[item ** i for i in range(order)] for item in lst])


def bv_to_br(b_v):
    return 0.4437 * b_v ** 2 + 1.2114 * b_v + 0.194


def abs_v_to_mass(abs_v):
    c0 = 0.19226
    c1 = -0.050737
    c2 = 0.010137
    c3 = -0.00075399
    c4 = -1.9858e-05
    x0 = 13.0

    term2 = c1 * (abs_v - x0)
    term3 = c2 * ((abs_v - x0) ** 2)
    term4 = c3 * ((abs_v - x0) ** 3)
    term5 = c4 * ((abs_v - x0) ** 4)
    return c0 + term2 + term3 + term4 + term5


def abs_k_to_mass(abs_k):
    c0 = 0.2311
    c1 = -0.1352
    c2 = 0.04
    c3 = 0.0038
    c4 = -0.0032
    x0 = 7.5

    term2 = c1 * (abs_k - x0)
    term3 = c2 * ((abs_k - x0) ** 2)
    term4 = c3 * ((abs_k - x0) ** 3)
    term5 = c4 * ((abs_k - x0) ** 4)
    return c0 + term2 + term3 + term4 + term5


rpo = np.linspace(-1, 9, 100)
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.invert_yaxis()


plt.scatter(rpo, abs_k_to_mass(rpo))
#%%

path = "D:/dev/spin_down/new_data/"

cluster_list = os.listdir(path)
files = [os.listdir(path + str(cluster)) for cluster in cluster_list]
ages = [50, 625.4, 13, 220, 150, 550, 130, 210, 150, 40, 1050, 125, 578, 11]
distances = [185, 47, 470, 1186, 1383, 1107, 130, 177, 145]
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
        "age": ages[i],
    }
    for i, file_list in enumerate(files)
]

for index, item in enumerate(dict_list):
    item.update({"author": files[index]})


#%%
color_chart = pd.read_csv("D:/dev/spin_down/colour_chart.csv", comment="#", sep=" ")

fit_vk_bv = LinearRegression()
fit_vk_bv.fit(color_chart.V_KS.to_numpy().reshape(-1, 1), color_chart.B_V.to_numpy())

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

# wright et al. 2018
wright_v_k_18 = np.array(
    [1.42, 1.64, 2.00, 2.54, 3.14, 3.53, 3.88, 4.37, 4.74, 5.14, 5.63]
)
wright_mass_18 = np.array(
    [1.17, 1.09, 0.95, 0.83, 0.70, 0.61, 0.50, 0.30, 0.22, 0.18, 0.14]
)
fit_vk_mass_18 = LinearRegression().fit(wright_v_k_18[:, np.newaxis], wright_mass_18)

fit_vk_mass = LinearRegression()
fit_vk_mass.fit(
    convert_order([np.average(item) for item in wright_v_k], 3),
    [np.average(item) for item in wright_mass],
)

fit_bv_mass = LinearRegression()
fit_bv_mass.fit(
    convert_order([np.average(item) for item in wright_b_v], 3),
    [np.average(item) for item in wright_mass],
)


b_v = np.linspace(0, 2, 50)
b_r = bv_to_br(b_v)

b_r = b_r[:, np.newaxis]
poly = PolynomialFeatures(2)
x = poly.fit_transform(b_r)

fit_br_bv = LinearRegression().fit(x, b_v)

z = np.linspace(0, 9, 20)
x = convert_order(z, 3)
# plt.scatter(
#     [np.average(item) for item in wright_v_k],
#     [np.average(item) for item in wright_mass],
# )
# plt.plot(z, fit_vk_mass.predict(x))

# plt.scatter(fit_br_bv.predict(poly.fit_transform(b_v[:, np.newaxis])), b_v)
#%% ALPHA PER
pm_list = [" "] * len(dict_list)
dict_item = [item for item in dict_list if item["name"] == "alpha_per"][0]

period = dict_item["data_frames"][0].Prot.to_numpy()
pm_list[0] = {
    "Per": dict_item["data_frames"][0].Prot.to_numpy(),
    "Mass": fit_bv_mass.predict(
        convert_order(dict_item["data_frames"][0]["B-V"].to_numpy(), 3)
    ),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[0]["Mass"], pm_list[0]["Per"])


#%% HYADES
dict_item = [item for item in dict_list if item["name"] == "hyades"][0]

vmag = dict_item["data_frames"][0].Vmag.to_numpy()
kmag = dict_item["data_frames"][0].Ksmag.to_numpy()
v_k = vmag - kmag
period_0 = dict_item["data_frames"][0].Per.to_numpy()[
    ~np.isnan(dict_item["data_frames"][0].Per.to_numpy())
][~np.isnan(v_k)]
v_k = v_k[~np.isnan(v_k)]


pm_list[1] = {
    "Per": np.append(
        period_0, dict_item["data_frames"][1].IPer.to_numpy()
    ),  # dict_item["data_frames"][0].Per.to_numpy(),
    "Mass": np.append(
        fit_vk_mass.predict(convert_order(v_k, 3)),
        dict_item["data_frames"][1].Mass.to_numpy(),
    ),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[1]["Mass"], pm_list[1]["Per"])
#%% h_per
dict_item = [item for item in dict_list if item["name"] == "h_per"][0]
dict_item["data_frames"][0].describe()

pm_list[2] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": dict_item["data_frames"][0].Mass,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[2]["Mass"], pm_list[2]["Per"])

#%% M34
dict_item = [item for item in dict_list if item["name"] == "m34"][0]

b_v = dict_item["data_frames"][0]["(B-V)0"].to_numpy()

period_0 = dict_item["data_frames"][0].Prot.to_numpy()[~np.isnan(b_v)]
b_v = b_v[~np.isnan(b_v)]
pm_list[3] = {
    "Per": period_0,
    "Mass": fit_bv_mass.predict(convert_order(b_v, 3)),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[3]["Mass"], pm_list[3]["Per"])

#%% M35
# dict_item["data_frames"][0]
dict_item = [item for item in dict_list if item["name"] == "m35"][0]

b_v = dict_item["data_frames"][0]["(B-V)0"].to_numpy()

period_0 = dict_item["data_frames"][0].Prot.to_numpy()[~np.isnan(b_v)]
b_v = b_v[~np.isnan(b_v)]

pm_list[4] = {
    "Per": period_0,
    "Mass": fit_bv_mass.predict(convert_order(b_v, 3)),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[4]["Mass"], pm_list[4]["Per"])

#%% M37
# dict_item["data_frames"][0].describe()
dict_item = [item for item in dict_list if item["name"] == "m37"][0]

# pm_list[5] = {
#     "Per": dict_item["data_frames"][0].Per.to_numpy(),
#     "Mass": fit_bv_mass.predict(
#         convert_order(
#             np.subtract(
#                 dict_item["data_frames"][0].Bmag.to_numpy(),
#                 dict_item["data_frames"][0].Vmag.to_numpy() + 0.2,
#             ),
#             3,
#         )
#     ),
#     "Name": dict_item["name"],
#     "Age": dict_item["age"],
# }
abs_vmag = app_to_abs(dict_item["data_frames"][0].Vmag.to_numpy(), 1383)

b_v = np.subtract(
    dict_item["data_frames"][0].Bmag.to_numpy(),
    dict_item["data_frames"][0].Vmag.to_numpy() + 0.2,
)

masses = []
for index, vmag in enumerate(abs_vmag):
    if vmag >= 9:
        masses.append(abs_v_to_mass(vmag))
    else:
        masses.append(fit_bv_mass.predict(convert_order([b_v[index]],3)))

pm_list[5] = {
    "Per": dict_item["data_frames"][0].Per.to_numpy(),
    "Mass": masses,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[5]["Mass"], pm_list[5]["Per"])

#%% m50
dict_item = [item for item in dict_list if item["name"] == "m50"][0]
dict_item["data_frames"][0].describe()

pm_list[6] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": dict_item["data_frames"][0].Mass,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[6]["Mass"], pm_list[6]["Per"])

#%% ngc2301
dict_item = [item for item in dict_list if item["name"] == "ngc2301"][0]
dict_item["data_frames"][0].describe()

b_r = dict_item["data_frames"][0]["B-R"].to_numpy()
b_v = fit_br_bv.predict(poly.fit_transform(b_r.reshape(-1, 1)))

pm_list[7] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": fit_bv_mass.predict(convert_order(b_v, 3)),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[7]["Mass"], pm_list[7]["Per"])

#%%  ngc2516
dict_item = [item for item in dict_list if item["name"] == "ngc2516"][0]
dict_item["data_frames"][0].describe()

pm_list[8] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": dict_item["data_frames"][0].Mass,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[8]["Mass"], pm_list[8]["Per"])


#%% ngc2547
dict_item = [item for item in dict_list if item["name"] == "ngc2547"][0]
dict_item["data_frames"][0].describe()


pm_list[9] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": dict_item["data_frames"][0].Mass,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}


fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[9]["Mass"], pm_list[9]["Per"])

#%% NGC6811
dict_item = [item for item in dict_list if item["name"] == "ngc6811"][0]

dict_item["data_frames"][0].describe()


pm_list[10] = {
    "Per": dict_item["data_frames"][0].Per,
    "Mass": dict_item["data_frames"][0].Mass,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[10]["Mass"], pm_list[10]["Per"])


#%% PLEIADES
dict_item = [item for item in dict_list if item["name"] == "pleiades"][0]

# print(dict_item["data_frames"][1].describe())


pm_list[11] = {
    "Per": dict_item["data_frames"][1].Prot.to_numpy(),
    "Mass": fit_vk_mass.predict(
        convert_order(dict_item["data_frames"][1]["(V-K)0"].to_numpy(), 3)
    ),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[11]["Mass"], pm_list[11]["Per"])

#%% PRAESEPE
dict_item = [item for item in dict_list if item["name"] == "praesepe"][0]

# print(dict_item["data_frames"][2].describe())

period_2 = dict_item["data_frames"][2].Per.to_numpy()

v_k = dict_item["data_frames"][2]["V-Ks"].to_numpy()[~np.isnan(period_2)]
app_k = dict_item["data_frames"][2]["Ksmag"].to_numpy()[~np.isnan(period_2)]
period_2 = period_2[~np.isnan(period_2)]


abs_kmag = app_to_abs(app_k, 177)

# masses = []

# for index, kmag in enumerate(abs_kmag):
#     if kmag >= 5.5:
#         masses.append(abs_k_to_mass(kmag))
#     else:
#         changed_v_k = PolynomialFeatures(2).fit_transform([[v_k[index]]])
#         masses.append(fit_vk_mass.predict(changed_v_k)[0])

masses = []
for index, kmag in enumerate(abs_kmag):
    if kmag >= 5.5:
        masses.append(abs_k_to_mass(kmag))
    else:

        masses.append(fit_vk_mass_18.predict([[v_k[index]]]))

pm_list[12] = {
    "Per": period_2,
    "Mass": masses,
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

# pm_list = np.append(
#     pm_list,
#     {
#         "Per": period_2,
#         "Mass": fit_vk_mass.predict(convert_order(v_k, 3)),
#         "Name": dict_item["name"],
#         "Age": dict_item["age"],
#     },
# )
# pm_list = np.append(
#     pm_list,
#     {
#         "Per": period_2,
#         "Mass": fit_vk_mass_18.predict(v_k[:,np.newaxis] - 0.2),
#         "Name": dict_item["name"],
#         "Age": dict_item["age"],
#     },
# )
fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[12]["Mass"], pm_list[12]["Per"])

#%% UPPER SCORPION
dict_item = [item for item in dict_list if item["name"] == "usco"][0]

# print(dict_item["data_frames"][0]["(V-Ks)0"].describe())

period_0 = dict_item["data_frames"][0].Per.to_numpy()

v_k = dict_item["data_frames"][0]["(V-Ks)0"].to_numpy()[~np.isnan(period_0)]

period_0 = period_0[~np.isnan(period_0)]

period_0 = period_0[~np.isnan(v_k)]
v_k = v_k[~np.isnan(v_k)]


pm_list[13] = {
    "Per": period_0,
    "Mass": fit_vk_mass.predict(convert_order(v_k, 3)),
    "Name": dict_item["name"],
    "Age": dict_item["age"],
}

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.invert_xaxis()
ax.set(title=dict_item["name"], xlim=(1.8, 0.0), ylim=(-2, 35))
ax.scatter(pm_list[13]["Mass"], pm_list[13]["Per"])


#%%
sorted_age_index = np.argsort([item["Age"] for item in pm_list])
fig, ax = plt.subplots(4, 4, figsize=(24, 16))

pm_list = np.array(pm_list)

for num, data in enumerate(pm_list[sorted_age_index]):
    mass = data["Mass"]
    period = data["Per"]

    r, c = divmod(num, 4)
    # print(r, c)

    ax[r][c].invert_xaxis()

    ax[r][c].plot(
        mass,
        period,
        linestyle="none",
        marker="x",
        label=data["Name"] + " " + str(data["Age"]),
    )
    ax[r][c].legend()
    ax[r][c].set(xlim=(1.7, 0.0), ylim=(-2, 40.0))  # ylim = (0, 100), yscale = "log",)


#%%
