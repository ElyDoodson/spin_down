#%%
import numpy as np
import os

from scipy.optimize import minimize, curve_fit, least_squares
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

plt.style.use("dark_background")

plt.style.use("ggplot")
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
        (abs(np.subtract(general_polynomial(parameters, x_true, b0), y_true))) ** 2
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


def chi_poly_fit(params, x_true, b0, coeffs):
    # alpha is the slope, sigma is the value the switch happens at
    alpha, sigma = params
    # print(alpha)
    # print(sigma)
    return [
        sum(
            [
                b0,
                sum(
                    np.array([coeffs[i] * x ** i for i in range(len(coeffs))])
                    * (1 / (1 + np.exp(-alpha * (x + sigma))))
                ),
            ]
        )
        for x in x_true
    ]


def chi_sum_residuals(parameters, x_true, y_true, b0, coeffs):
    return sum((np.subtract(y_true, chi_poly_fit(parameters, x_true, b0, coeffs))) ** 2)


def root_general_polynomial(parameters, x_true, b0):
    return np.array(
        [
            sum([b0, np.prod([x - parameters[i] for i in range(len(parameters))])])
            for x in x_true
        ]
    )


def root_sum_residuals(parameters, x_true, y_true, b0):
    return sum(
        (abs(np.subtract(root_general_polynomial(parameters, x_true, b0), y_true))) ** 2
    )


# chi_poly_fit([1, 2], [1], 0, [3, 1, 1, 1])

root_general_polynomial([1, 1, 1], [1], 0)

#%% Data importation and Cluster Dictionary Initialisation
path = "C:/dev/spin_down/mistmade_data/"
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
    if name == "m37":
        boolean_mask = cluster_dict[name]["df"].Mass < 1.13
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

# Dropping out of bound clusters Hyades and Usco
del cluster_dict["alpha_per"]
del cluster_dict["h_per"]
del cluster_dict["usco"]

# Deleting low data clusters
del cluster_dict["ngc2547"]
del cluster_dict["hyades"]

# %% Declaring name lists and dropping values
# redeclaring names after removal
names = np.array(list(cluster_dict.keys()))

# list of names sorted by age
sorted_names = names[np.argsort([cluster_dict[name]["age"] for name in names])]

# Dropping fast rotators
def line(x, b0=26.0, b1=-32.2):
    return b0 + b1 * x


praesepe_df = cluster_dict["praesepe"]["df"]
m37_df = cluster_dict["m37"]["df"]

for name in sorted_names:
    # Removes data below line (fast rotators)
    dframe = cluster_dict[name]["df"]
    fast_rotators = dframe.index[dframe.Per < line(dframe.Mass)]
    cluster_dict[name]["df"] = dframe.drop(fast_rotators, axis=0)
    if name == "praesepe":
        praesepe_line_df = cluster_dict[name]["df"]
    if name == "m37":
        m37_line_df = cluster_dict[name]["df"]

    df = cluster_dict[name]["df"]
    df = df[~np.isnan(df.Per)]
    df = df[~np.isnan(df.Mass)]

    labels, bins = pd.cut(df.Mass, 10, labels=False, retbins=True)

    unique_labels = np.unique(labels)

    binned_df = [df[labels == label] for label in unique_labels]
    scores = [np.abs(stats.zscore(df.Per.to_numpy())) for df in binned_df]
    # zscore
    trimmed_df = [df[scores[ind] < 1.0] for ind, df in enumerate(binned_df)]
    cluster_dict[name]["df"] = pd.concat(trimmed_df)

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(xlim=(1.5, 0), ylim=(-1, 35), xlabel="Mass (M_Solar)", ylabel="Period (Days)")
ax.scatter(praesepe_df.Mass, praesepe_df.Per, c="orange", s=3.5, label="Below line")

ax.scatter(
    praesepe_line_df.Mass,
    praesepe_line_df.Per,
    c="brown",
    s=3.5,
    label="90% Deleted Data",
)
ax.scatter(
    cluster_dict["praesepe"]["df"].Mass,
    cluster_dict["praesepe"]["df"].Per,
    c="green",
    s=3.5,
)

ax.legend()

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.set(xlim=(1.5, 0), ylim=(-1, 35), xlabel="Mass (M_Solar)", ylabel="Period (Days)")
ax.scatter(m37_df.Mass, m37_df.Per, c="orange", s=3.5, label="90% Deleted Data")
ax.scatter(m37_line_df.Mass, m37_line_df.Per, c="brown", s=3.5, label="Below line")

ax.scatter(
    cluster_dict["m37"]["df"].Mass, cluster_dict["m37"]["df"].Per, c="green", s=3.5
)
ax.legend()

#%% Cluster Graphs and Tables
# fig size is nice for 24,18. Reduced to reduce run time.

# Data figure
fig, ax = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.03, hspace=0.05)

# # Coefficient Figure
# fig3, ax3 = plt.subplots(2, 3, figsize=(18, 12), sharex=True)
# fig3.subplots_adjust(wspace=0.03, hspace=0.05)

coeff_list = []
for num, name in enumerate(sorted_names):
    r, c = divmod(num, 3)
    # print(r, c)

    # Removing Nans(breaks fitting)
    df = cluster_dict[name]["df"]
    df = df[~np.isnan(df.Per)]
    df = df[~np.isnan(df.Mass)]

    mass = df["Mass"]
    period = df["Per"]

    # Plotting cluster data
    ax[r][c].scatter(
        mass,
        period,
        # linestyle="none",
        marker="x",
        label=("%s %iMYrs") % (name, cluster_dict[name]["age"]),
        s=1.5,
        c="#189ad3",
    )
    # Setting initial Parameters
    b0 = 0.46
    # optimised on praesepe
    # starting_coeffs = [
    #     -33.01648156,
    #     406.80460616,
    #     -936.89032886,
    #     500.33705448,
    #     685.35829236,
    #     -887.19870956,
    # 274.57382769,
    # -36.96156584,
    # ]

    # starting_coeffs = [
    #     76.95610948,
    #     -108.71350769,
    #     -224.85132865,
    #     645.56928067,
    #     -504.76986033,
    #     125.74607982,
    # ]

    # ##The one to tweak
    # starting_coeffs = [
    #     52.57361218,
    #     -131.49428603,
    #     131.07679869,
    #     -18.61060615,
    #     -29.63407946,
    #     5.77416665,
    # ]
    starting_coeffs = [
        41.8919926639974,
        -29.44378826,
        -85.14549312,
        136.82333888,
        -55.33518376,
    ]
    coeffs = minimize(
        sum_residuals,
        starting_coeffs,
        args=(mass, period, b0),
        tol=1e-10,
        bounds=[[coeff - 5, 5 + coeff] for coeff in starting_coeffs[:]],
        # bounds=[  [None, None], [None, None],[1.3,1.3], [None, None], [None, None]],
    )
    coeff_list.append(coeffs.x)

    cluster_dict[name].update({"coefficients": coeffs.x})

    z = np.linspace(1.4, 0.0, 200)
    ax[r][c].plot(
        z,
        general_polynomial(coeffs.x, z, b0),
        label="Pred from minimised coeffs",
        # linewidth=3.5,
        # c="purple",
    )

    # ax[r][c].plot(z, [b0] * len(z))
    ax[r][c].legend()
    ax[r][c].set(xlim=(1.5, -0.1), ylim=(-2, 35.0))

    # coeff_strings = ["b{}".format(i + 1) for i in range(len(coeffs.x))]

    # ax3[r][c].bar(
    #     x=np.arange(len(coeff_strings)),
    #     height=np.subtract(coeffs.x, starting_coeffs),
    #     tick_label=coeff_strings,
    # )


# # print("------------------------------------------------------------")
# # print(
# #     "{:<13s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:>8s}".format(
# #         "name", "b2", "b3", "b4", "b5/b4", "b6", "b7", "b8", "b5"
# #     )
# # )
# # print("------------------------------------------------------------")
# # for i, item in enumerate(coeff_list):
# #     print(
# #         "{:<13s}{:^8.1f}{:^8.1f}{:^8.1f}{:^8.2f}{:^8.1f}{:^8.1f}{:>8.1f}{:>8.1f}".format(
# #             sorted_names[i], *item, -item[3] * item[4]
# #         )
# #     )
# fig.show()
# fig3.show()


# #%% Boxplots
# fig, ax = plt.subplots(1, figsize=(11.5, 7))

# draw_boxplot(
#     [[item[i] for item in coeff_list] for i in range(len(coeffs.x))], "green", "black"
# )
# plt.xticks([1, 2, 3, 4, 5], ["b2", "b3", "b4", "b5", "b6"])

# #%% Coefficient Plots
# fig2, ax2 = plt.subplots(1, figsize=(11.5, 7))

# ax2.scatter(
#     [[cluster_dict[name]["age"]] * len(coeffs.x) for name in sorted_names], coeff_list
# )
# ax2.plot(
#     [cluster_dict[name]["age"] for name in sorted_names],
#     [item[0] for item in coeff_list],
# )
# fig2.show()


#%% RIDGE REGRESSION
#
#
#
#
#
#
#

name = "praesepe"

df = cluster_dict[name]["df"]

if name == "praesepe":
    df = df[df.Mass > 0.44]
# if name == "m37":
#     df = df[df.Mass > 0.5]


df = df[~np.isnan(df.Per)]
df = df[~np.isnan(df.Mass)]


mass = df.Mass.to_numpy()
period = df.Per.to_numpy()
# if name == "m37":
#     mass = mass + 0.1

alphas = np.logspace(-10, 3, 100)
degrees = np.arange(3, 4)

model_df = pd.DataFrame(columns=["degree", "mse", "lambda"])
for degree in degrees:

    kf = KFold(n_splits=10)
    kf.get_n_splits(mass)

    mse_list = []
    for alpha in alphas:
        lr = Ridge(alpha)
        poly = PolynomialFeatures(degree)

        kfold_mse = np.array([])
        for train_index, test_index in kf.split(mass):
            X_train, X_test = mass[train_index], mass[test_index]
            y_train, y_test = period[train_index], period[test_index]

            X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
            X_test_poly = poly.fit_transform(X_test[:, np.newaxis])

            lr.fit(X_train_poly, y_train)

            mse = mean_squared_error(y_test, lr.predict(X_test_poly))
            kfold_mse = np.append(kfold_mse, mse)
        # print(kfold_mse)
        mse_list.append(kfold_mse.mean())
    model_df = model_df.append(
        {
            "degree": degree,
            "mse": min(mse_list),
            "lambda": alphas[mse_list.index(min(mse_list))],
        },
        ignore_index=True,
    )

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.set(
    xlim=(1.5, 0), ylim=(-1, 35), xlabel=r"Mass ($M_\odot$)", ylabel="Period ($days$)"
)
ax.scatter(mass, period, c="#189ad3", s=3.5)
ax.annotate(
    "Degree = {:.2f}\nLambda = {:.4f}\nMSE = {:.4f}".format(
        model_df["degree"][model_df.mse.idxmin()],
        model_df["lambda"][model_df.mse.idxmin()],
        model_df["mse"][model_df.mse.idxmin()],
    ),
    xy=(0.4, 5),
    color="#189ad3",
)

lr = Ridge(alpha=model_df["lambda"][model_df.mse.idxmin()])
poly = PolynomialFeatures(int(model_df["degree"][model_df.mse.idxmin()]))


X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
X_test_poly = poly.fit_transform(X_test[:, np.newaxis])

lr.fit(np.vstack((X_test_poly, X_train_poly)), np.hstack((y_test, y_train)))

white_space = np.linspace(1.5, 0, 100)
ax.plot(white_space, lr.predict(poly.fit_transform(white_space[:, np.newaxis])))

#%% MINIMISATION
#
#
#
#
#
lr = LinearRegression()
deg = 3
poly = PolynomialFeatures(deg)

df = cluster_dict["praesepe"]["df"]
df = df[df.Mass > 0.46]
df = df[df.Mass < 1.29]
df = df[~np.isnan(df.Per)]
df = df[~np.isnan(df.Mass)]

# filter_ = (df.Mass > 1.0) & (df.Per > 4.5)
# filter_ = ~filter_

# df = df[filter_]


mass = df.Mass.to_numpy()
period = df.Per.to_numpy()
poly_mass = poly.fit_transform(mass.reshape(-1, 1))
# lr.fit(poly_mass, period)

fun = lambda p: np.sum((np.sum(p * poly_mass, axis=-1) - period) ** 2)

res = minimize(
    fun,
    np.full([deg + 1], 0),
    # [0, 0, 0, -20],
    # bounds=[[None, None], [None, None], [None, None], [None, None]],
)
best_p = res.x

pred = lambda x: np.sum(x * best_p, axis=-1)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.set(
    xlim=(1.5, 0), ylim=(-1, 35), xlabel=r"Mass ($M_\odot$)", ylabel="Period ($days$)"
)
ax.scatter(mass, period, c="green", s=3.5)

# ax.scatter(cluster_dict["praesepe"]["df"].Mass, cluster_dict["praesepe"]["df"].Per, c="green", s=3.5)


white_space = np.linspace(1.5, 0, 100)
white_space_p = poly.transform(white_space.reshape(-1, 1))
ax.plot(white_space, [pred(x) for x in white_space_p])


# %% TEST CELL
