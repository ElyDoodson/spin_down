#%%
import numpy as np
import os

# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics.scorer import make_scorer
# from sklearn.svm import SVR
from scipy.optimize import minimize, curve_fit, least_squares
import pandas as pd

from matplotlib import pyplot as plt

#%%
def general_polynomial(parameters, x_true, b0):
    return np.array(
        [
            sum([b0, sum([parameters[i] * x ** i for i in range(len(parameters))])])
            for x in x_true
        ]
    )


def sum_residuals(parameters, x_true, y_true, b0):
    return sum(
        (abs(np.subtract(general_polynomial(parameters, x_true, b0), y_true))) ** 1
    )


def y_pred(m, b0, b2, b3, b4, b5):
    return b0 + (b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5))))


def calculate_chi(m, b4, b5):
    return 1 / (1 + np.exp(-(b4 * m + b5)))


def draw_boxplot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)
    plt.setp(bp["boxes"], facecolor=fill_color)
    plt.setp(bp["fliers"], markeredgecolor=edge_color)


# # linear

# def y_pred_test(params, m):
#     b0, b2, b3, b4, b5 = params
#     return b0 + (b2 * m + b3 ) * (1 / (1 + np.exp(-(b4 * m + b5))))


# def mse(params, m, period):
#     b0, b2, b3, b4, b5 = params

#     return np.sum(
#         (b0 + (b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5)))) - period) ** 2
#     ) / len(period)


# # Quadratic


# def y_pred_test(params, m):
#     b0, b2, b3, b4, b5, b6 = params
#     return b0 + (b6 * m ** 2 + b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5))))


# def mse(params, m, period):
#     b0, b2, b3, b4, b5, b6 = params

#     return np.sum(
#         (b0 + (b6 * m ** 2 + b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5)))) - period)
#         ** 2
#     ) / len(period)

# # Quadratic, fixed b0


# def y_pred_test(params, m, b0):
#     b2, b3, b4, b5, b6, b7 = params
#     return b0 + (b7 * m ** 3 + b6 * m ** 2 + b2 * m + b3) * (
#         1 / (1 + np.exp(-(b4 * m + b5 * b4)))
#     )


# def mse(params, m, period, b0):
#     b2, b3, b4, b5, b6, b7 = params

#     return np.sum(
#         (
#             b0
#             + (b7 * m ** 3 + b6 * m ** 2 + b2 * m + b3)
#             * (1 / (1 + np.exp(-(b4 * m + b5 * b4))))
#             - period
#         )
#         ** 2
#     ) / len(period)

##pentanic, fixed b0


def y_pred_test(params, m, b0):
    b2, b3, b4, b5, b6, b7, b8 = params
    return b0 + (b8 * m ** 4 + b7 * m ** 3 + b6 * m ** 2 + b2 * m + b3) * (
        1 / (1 + np.exp(-(b4 * m + b5 * b4)))
    )


def mse(params, m, period, b0):
    b2, b3, b4, b5, b6, b7, b8 = params

    return np.sum(
        (
            b0
            + (b8 * m ** 4 + b7 * m ** 3 + b6 * m ** 2 + b2 * m + b3)
            * (1 / (1 + np.exp(-(b4 * m + b5 * b4))))
            - period
        )
        ** 2
    ) / len(period)


#%% Cluster Dictionary
path = "D:/dev/spin_down/mistmade_data/"
files = os.listdir(path)
cluster_list = [name[:-4] for name in files]
cluster_dict = {}
names = np.array([name[:-16] for name in cluster_list])

# Makes dictionary with name access to cluster, containing "age" and "df"
# which is a pd.df that contains period "Per" and mass "Mass"
for i, cluster in enumerate(cluster_list):
    name = cluster[:-16]
    # print(name)
    age = float(cluster[-12:])
    # print(age)
    cluster_dict.update(
        {name: {"age": age, "df": pd.read_csv(path + files[i], sep="\t")}}
    )

# Limits the maximum mass to 1.3, where a convective envelope stops
for name in names:
    boolean_mask = cluster_dict[name]["df"].Mass < 1.3
    cluster_dict[name]["df"].Per = cluster_dict[name]["df"].Per.loc[boolean_mask]
    cluster_dict[name]["df"].Mass = cluster_dict[name]["df"].Mass.loc[boolean_mask]

# Merging Pleiades + m50, m35  + ngc2516, ngc2301 + m34
cluster_dict.update(
    {
        "m50+pleiades": {
            "age": 127,
            "df": pd.concat(
                [cluster_dict["pleiades"]["df"], cluster_dict["m50"]["df"]]
            ),
        }
    }
)
cluster_dict.update(
    {
        "m35+ngc2516": {
            "age": 149,
            "df": pd.concat([cluster_dict["m35"]["df"], cluster_dict["ngc2516"]["df"]]),
        }
    }
)
cluster_dict.update(
    {
        "m34+ngc2301": {
            "age": 215,
            "df": pd.concat([cluster_dict["m34"]["df"], cluster_dict["ngc2301"]["df"]]),
        }
    }
)
# Deleting the clusters that were merged
del cluster_dict["pleiades"]
del cluster_dict["m50"]
del cluster_dict["m35"]
del cluster_dict["ngc2516"]
del cluster_dict["m34"]
del cluster_dict["ngc2301"]

# Dropping useless clusters Hyades and Usco
del cluster_dict["alpha_per"]
del cluster_dict["usco"]

#%% Declaring name lists and dropping values
# redeclaring names after removal
names = np.array(list(cluster_dict.keys()))

# list of names sorted by age
sorted_names = names[np.argsort([cluster_dict[name]["age"] for name in names])]

# Dropping fast rotators
def line(x, b0=17.0, b1=-16.2):
    return b0 + b1 * x


for name in sorted_names:

    dframe = cluster_dict[name]["df"]

    ## Shows Before
    # fig, ax = plt.subplots(1, figsize=(11.5, 7))
    # ax.set(xlim = (1.4,0), ylim = (-2,35))
    # ax.scatter(dframe.Mass.to_numpy(), dframe.Per.to_numpy())

    fast_rotators = dframe.index[dframe.Per < line(dframe.Mass)]
    cluster_dict[name]["df"] = dframe.drop(fast_rotators, axis=0)

    ## And after of the reduces data 
    # fig2, ax2 = plt.subplots(1, figsize=(11.5, 7))
    # ax2.set(xlim = (1.4,0), ylim = (-2,35))
    # ax2.scatter(dframe.Mass.to_numpy(), dframe.Per.to_numpy())


#%% Cluster Graphs and Tables
# fig size is nice for 24,18. Reduced to reduce run time.

# Data figure
fig, ax = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.03, hspace=0.05)

# Coefficient Figure
fig3, ax3 = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
fig3.subplots_adjust(wspace=0.03, hspace=0.05)

coeff_list = []
for num, name in enumerate(sorted_names):
    r, c = divmod(num, 3)
    # print(r, c)
    df = cluster_dict[name]["df"]

    df = df[~np.isnan(df.Per)]
    df = df[~np.isnan(df.Mass)]

    # mass = cluster_dict[name]["df"]["Mass"]
    # period = cluster_dict[name]["df"]["Per"]
    mass = df["Mass"]
    period = df["Per"]

    ax[r][c].plot(
        mass,
        period,
        linestyle="none",
        marker="x",
        label=("%s %iMYrs") % (name, cluster_dict[name]["age"]),
        markersize=1.5,
    )
    b0 = 0.46
    # starting_coeffs = [-120, 45, 85, -0.25, 54, 48, -42]
    starting_coeffs = [
        -33.01648156,
        406.80460616,
        -936.89032886,
        500.33705448,
        685.35829236,
        -887.19870956,
        274.57382769,
        -36.96156584,
    ]
    coeffs = minimize(
        sum_residuals, starting_coeffs, args=(mass, period, b0), tol=1e-16
    )
    coeff_list.append(coeffs.x)

    z = np.linspace(1.4, 0.0, 200)

    ax[r][c].plot(
        z,
        general_polynomial(coeffs.x, z, b0),
        label="Pred from minimised coeffs",
        linewidth=5,
        c="purple",
    )
    ax[r][c].plot(z, [b0] * len(z))
    ax[r][c].legend()
    ax[r][c].set(xlim=(1.5, -0.1), ylim=(-2, 25.0))

#     coeff_strings = ["b2", "b3", "b4", "b5/b4", "b6", "b7", "b8"]

#     ax3[r][c].bar(
#         x=np.arange(len(coeff_strings)), height=coeffs.x, tick_label=coeff_strings
#     )

# fig.show()
# print("------------------------------------------------------------")
# print(
#     "{:<13s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:>8s}".format(
#         "name", "b2", "b3", "b4", "b5/b4", "b6", "b7", "b8", "b5"
#     )
# )
# print("------------------------------------------------------------")
# for i, item in enumerate(coeff_list):
#     print(
#         "{:<13s}{:^8.1f}{:^8.1f}{:^8.1f}{:^8.2f}{:^8.1f}{:^8.1f}{:>8.1f}{:>8.1f}".format(
#             sorted_names[i], *item, -item[3] * item[4]
#         )
#     )

fig3.show()


#%% Boxplots
fig, ax = plt.subplots(1, figsize=(11.5, 7))

draw_boxplot(
    [[item[i] for item in coeff_list] for i in range(len(coeffs.x))], "green", "black"
)
plt.xticks([1, 2, 3, 4, 5], ["b2", "b3", "b4", "b5", "b6"])

#%% Coefficient Plots
fig2, ax2 = plt.subplots(1, figsize=(11.5, 7))

ax2.scatter(
    [[cluster_dict[name]["age"]] * len(coeffs.x) for name in sorted_names], coeff_list
)
ax2.plot(
    [cluster_dict[name]["age"] for name in sorted_names],
    [item[0] for item in coeff_list],
)
fig2.show()


#%%


df = cluster_dict["ngc6811"]["df"]

df = df[~np.isnan(df.Per)]
df = df[~np.isnan(df.Mass)]

# sum_residuals([2, 3, 5], [1, 2, 3], [10, 28, 56], 0.46)
starting_coeffs = [
    -33.01648156,
    406.80460616,
    -936.89032886,
    500.33705448,
    685.35829236,
    -887.19870956,
    274.57382769,
    -36.96156584,
]
# starting_coeffs = [0,0,0,0,0,0,0,0]
values = minimize(
    sum_residuals, starting_coeffs, args=(df.Mass, df.Per, 0.46), tol=1e-16
)

print(values.x)
print("Sum_res = {}".format(sum_residuals(values.x, df.Mass, df.Per, 0.46)))

fig2, ax2 = plt.subplots(1, figsize=(11.5, 7))
ax2.set(xlim=(1.4, 0.0), ylim=(-1, 35.0))
z = np.linspace(-2, 2, 200)
ax2.scatter(df.Mass, df.Per, c="#ff8c00")
ax2.plot(z, general_polynomial(values.x, z, 0.46), c="green", linewidth=3.5)

#%%
coeffs_praesepe = [
    -59.96209914,
    490.32021082,
    -963.9578435,
    436.3633149,
    663.02484292,
    -848.7525428,
    328.32736006,
    -36.96156584,
]

coeffs_6811 = [
    -3.30164816e01,
    4.06804606e02,
    -9.36890329e02,
    5.00337054e02,
    6.85358292e02,
    -8.87198710e02,
    2.74573828e02,
    -4.27375949,
]

coeffs_m37 = [
    -30.74459788,
    333.02087399,
    -768.31105221,
    464.19979473,
    546.46521987,
    -899.4099719,
    412.10128712,
    -53.15257031,
]

fig, ax = plt.subplots(1, figsize=(11.5, 7))
coeff_strings = ["b{}".format(str(i)) for i in range(len(coeffs_praesepe))]

x = np.arange(2 * len(coeff_strings), step=2)
width = 0.3
ax.bar(
    x - 1.5 * width,
    starting_coeffs,
    width=width,
    color="white",
    align="center",
    label="Starting coeffs",
)
ax.bar(x - 0.5 * width, coeffs_m37, label="m37", width=width, color="c", align="center")
ax.bar(
    x + 0.5 * width,
    coeffs_praesepe,
    label="praesepe",
    width=width,
    color="m",
    align="center",
)
ax.bar(
    x + 1.5 * width,
    coeffs_6811,
    label="ngc6811",
    width=width,
    color="orange",
    align="center",
)
ax.legend()
