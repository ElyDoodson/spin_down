#%%
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import PowerNorm, Normalize

from scipy import stats

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


#%% FUNCTIONS
def sigmoid(x, a=1, b=0):
    return np.divide(1, 1 + np.exp(-(a * x + b)))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


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


def general_polynomial_chi(parameters_all, x_true, b0):
    parameters = parameters_all[2:]
    alpha, beta = parameters_all[:2]
    return np.array(
        [
            sum(
                [
                    b0,
                    sum([parameters[i] * x ** i for i in range(len(parameters))])
                    * np.divide(1, 1 + np.exp(-(x * alpha + beta))),
                ]
            )
            for x in x_true
        ]
    )


def sum_residuals_chi(parameters, x_true, y_true, b0):
    return sum(
        (abs(np.subtract(general_polynomial_chi(parameters, x_true, b0), y_true))) ** 2
    )


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


def line(x, b0=26.0, b1=-32.2):
    return b0 + b1 * x


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
        boolean_mask = cluster_dict[name]["df"].Mass < 1.16
    cluster_dict[name]["df"].Per = cluster_dict[name]["df"].Per.loc[boolean_mask]
    cluster_dict[name]["df"].Mass = cluster_dict[name]["df"].Mass.loc[boolean_mask]

# Dropping out of bound clusters  Usco

del cluster_dict["usco"]

#%% Image of all clusters

# redeclaring names after removal
names = np.array(list(cluster_dict.keys()))

# list of names sorted by age
sorted_names = names[np.argsort([cluster_dict[name]["age"] for name in names])]
plt.style.use("ggplot")
# plt.style.use("fast")
fig, ax = plt.subplots(
    5,
    3,
    figsize=(8, 10),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.07, "wspace": 0.05},
    dpi=800,
)
# fig.subplots_adjust(wspace=0, hspace=0.0)

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
        marker="o",
        label=("%s ~%iMYrs") % (name, cluster_dict[name]["age"]),
        s=1,
        c="#189ad3",
    )
    ax[r][c].set(xlim=(1.4, -0.1), ylim=(-3, 45))
    ax[r][c].legend()

# ax[1][0].set(ylabel="Period ($days$)")
# ax[-1][1].set(xlabel=r"Mass ($M_\odot$)")

fig.text(0.5, 0.075, r"Mass ($M_\odot$)", ha="center", fontsize=15, color="#4C4C4C")
fig.text(
    0.05,
    0.5,
    "Period ($days$)",
    va="center",
    rotation="vertical",
    fontsize=15,
    color="#4C4C4C",
)

fig.savefig(
    "C:/Users/elydo/Documents/Harvard/Midterm_report_images/allclusters.png",
    bbox_inches="tight",
    dpi=1600,
)
#%% Cluster Merging and removal
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

# Deleting low data clusters
del cluster_dict["ngc2547"]
del cluster_dict["hyades"]
del cluster_dict["alpha_per"]

# dropping distracting cliusters
del cluster_dict["m34+ngc2301"]
del cluster_dict["ngc6819"]
del cluster_dict["m67"]
# %% Declaring name lists and dropping values

# redeclaring names after removal
names = np.array(list(cluster_dict.keys()))

# list of names sorted by age
sorted_names = names[np.argsort([cluster_dict[name]["age"] for name in names])]

#%% PLOTTING ALL CLUSTERS

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

plt.style.use("ggplot")
# plt.style.use("fast")
fig, ax = plt.subplots(
    2,
    3,
    figsize=(10, 5),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.07, "wspace": 0.05},
    dpi=200,
)
# fig.subplots_adjust(wspace=0, hspace=0.0)

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
        marker="o",
        label=("%s ~%iMYrs") % (name, cluster_dict[name]["age"]),
        s=3.5,
        c="#189ad3",
    )
    ax[r][c].set(xlim=(1.4, -0.1), ylim=(-1, 35))
    ax[r][c].legend()

# ax[1][0].set(ylabel="Period ($days$)")
# ax[-1][1].set(xlabel=r"Mass ($M_\odot$)")

fig.text(0.5, 0.05, r"Mass ($M_\odot$)", ha="center", fontsize=15, color="#4C4C4C")
fig.text(
    0.075,
    0.5,
    "Period ($days$)",
    va="center",
    rotation="vertical",
    fontsize=15,
    color="#4C4C4C",
)
ax[0][0].text(0.25, 16, "$P_i$", ha="center", color="#4C4C4C", fontsize=12)
ax[0][0].annotate(
    "",
    (0.25, -1),
    (0.25, 14),
    arrowprops={"arrowstyle": "|-|", "color": "#4C4C4C", "lw": 1},
    color="#4C4C4C",
    fontsize=8,
)
fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/plot1.png",
    bbox_inches="tight",
    dpi=800,
)

#%% PLOTTING SINGULAR CLUSTER

plt.rcParams.update({"font.size": 18})

fig, ax = plt.subplots(1, figsize=(10, 5), dpi=400)

mass = cluster_dict["praesepe"]["df"]["Mass"]
period = cluster_dict["praesepe"]["df"]["Per"]

ax.scatter(
    mass,
    period,
    marker="o",
    label=("%s ~%iMYrs") % (name, cluster_dict[name]["age"]),
    s=25,
    c="#189ad3",
)
ax.set(
    xlim=(1.4, -0.1),
    ylim=(-1, 35),
    ylabel="Period ($days$)",
    xlabel=r"Mass ($M_\odot$)",
)
ax.annotate(
    "Skumanich law for spin-down\n\t\t(~$t^{ - \\frac{1}{2}}$)",
    (0.65, 15),
    (1.2, 25),
    arrowprops={"arrowstyle": "->", "color": "#4C4C4C", "lw": 2.5},
    va="center",
    fontsize="19",
    color="#4C4C4C",
)
ax.annotate(
    "?",
    (0.5, 2.5),
    (0.1, 12),
    arrowprops={"arrowstyle": "->", "color": "#4C4C4C", "lw": 2.5},
    va="center",
    fontsize="25",
    color="#4C4C4C",
)
ax.annotate(
    " ",
    (1.0, 2.5),
    (0.1, 12),
    arrowprops={"arrowstyle": "->", "color": "#4C4C4C", "lw": 2.5},
    va="center",
    fontsize="25",
    color="#4C4C4C",
)


fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/plot2.png",
    dpi=800,
    bbox_inches="tight",
    # transparent=True,
    # facecolor="#E5E5E5",
)


# %%
# Reducing the data of the Clusters to 85%
for name in sorted_names:
    # Removes data below line (fast rotators)
    dframe = cluster_dict[name]["df"]
    fast_rotators = dframe.index[dframe.Per < line(dframe.Mass)]
    cluster_dict[name]["df_reduced"] = dframe.drop(fast_rotators, axis=0)
    if name == "praesepe":
        praesepe_line_df = cluster_dict[name]["df"]
    if name == "m37":
        m37_line_df = cluster_dict[name]["df"]

    df = cluster_dict[name]["df_reduced"]
    df = df[~np.isnan(df.Per)]
    df = df[~np.isnan(df.Mass)]

    labels, bins = pd.cut(df.Mass, 15, labels=False, retbins=True)

    unique_labels = np.unique(labels)

    binned_df = [df[labels == label] for label in unique_labels]
    scores = [np.abs(stats.zscore(df.Per.to_numpy())) for df in binned_df]
    trimmed_df = [df[scores[ind] < 1.44] for ind, df in enumerate(binned_df)]
    cluster_dict[name]["df_reduced"] = pd.concat(trimmed_df)

name = "praesepe"

df = cluster_dict[name]["df_reduced"]

if name == "praesepe":

    df = df[df.Mass > 0.44]

df = df[~np.isnan(df.Per)]
df = df[~np.isnan(df.Mass)]


mass = df.Mass.to_numpy()
period = df.Per.to_numpy()
if name == "m37":
    mass = mass + 0.1

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


mass_full = cluster_dict[name]["df"].Mass
period_full = cluster_dict[name]["df"].Per

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.set(
    xlim=(1.5, 0), ylim=(-1, 35), xlabel=r"Mass ($M_\odot$)", ylabel="Period ($days$)"
)
ax.scatter(mass_full, period_full, c="#189ad3", s=3.5)
# ax.annotate(

#     "Degree = {:.2f}\nLambda = {:.4f}\nMSE = {:.4f}".format(
#         model_df["degree"][model_df.mse.idxmin()],
#         model_df["lambda"][model_df.mse.idxmin()],
#         model_df["mse"][model_df.mse.idxmin()],
#     ),
#     xy=(0.4, 5),
#     color="#189ad3",
# )
ax.text(1.10, 9, "SLOW", rotation=23, color="#E15A45")

ax.text(1.07, 2, "FAST", rotation=0, color="#E15A45")

ax.text(0.62, 6, "TRANSITION", rotation=20, color="#4C4C4C")

lr = Ridge(alpha=model_df["lambda"][model_df.mse.idxmin()])
poly = PolynomialFeatures(int(model_df["degree"][model_df.mse.idxmin()]))


X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
X_test_poly = poly.fit_transform(X_test[:, np.newaxis])

lr.fit(np.vstack((X_test_poly, X_train_poly)), np.hstack((y_test, y_train)))

white_space = np.linspace(1.5, 0, 200)
ax.plot(white_space, lr.predict(poly.fit_transform(white_space[:, np.newaxis])))
ax.plot(white_space, [0.86] * len(white_space), color="#E15A45")

fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/slow_fast_transition.png",
    dpi=800,
    bbox_inches="tight",
    # transparent=True,
    # facecolor="#E5E5E5",
)
# %% Fast/Slow Transition for Static Time
fig, ax = plt.subplots(1, figsize=(10, 2))
line_norm = Normalize(-23, -3)
offset_data = np.arange(-23, -8, 0.5)

cmap = get_cmap("Blues")

for offset in sorted(offset_data):
    data = sigmoid(white_space, -20, -offset)

    norm_value = line_norm(offset)
    col = cmap(norm_value)

    ax.plot(white_space, data, lw=1.5, c=col)  # , label=offset)

# plotting one orange line to highlight clusters place in the plot
data = sigmoid(white_space, -20, 12)
ax.plot(white_space, data, lw=1.5, c="#1BE7FF")


data = sigmoid(white_space, -20, 15)
ax.plot(white_space, data, lw=1.5, c="#FFBC3F")


data = sigmoid(white_space, -20, 17)
ax.plot(white_space, data, lw=1.5, c="#E15A45")


ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(["$FAST$", "$TRANSITION$", "$SLOW$"])
ax.set_xticklabels([])
ax.set(xlabel=r"($\longleftarrow$ Increasing) Mass for Fixed Time")
# ax.set_title(

#     r"$\updownarrow$ Number of Stars in Regime",
#     color="#4C4C4C",
#     fontsize=22,
#     loc="left",
# )
# eta_line = np.linspace(0.419,1.14,100)
# ax.plot(eta_line, [0.5]*len(eta_line), color="#636363")
ax.annotate(
    "Initial Period\n($P_i$)Distribution",
    (1.16, 0.5),
    (-0.04, 0.32),
    arrowprops={"arrowstyle": "|-|", "color": "#4C4C4C", "lw": 1},
    color="#4C4C4C",
    fontsize=20,
)
fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/fixedtime_transition.png",
    dpi=800,
    bbox_inches="tight",
    # transparent=True,
    # facecolor="#E5E5E5",
)
#%% FITTED PLOT
coefficients_1 = minimize(
    sum_residuals_chi,
    [0, 0, 0, 0, 0, 0],
    args=(mass_full, period_full, 0.46),
    bounds=[
        [90, 90],
        [-35, -35],
        [66.69405999290854, 66.69405999290854],
        [-158.62725034, -158.62725034],
        [157.03891041, 157.03891041],
        [-56.08440611, -56.08440611],
    ],
)
coefficients_2 = minimize(
    sum_residuals_chi,
    [0, 0, 0, 0, 0, 0],
    args=(mass_full, period_full, 0.46),
    bounds=[
        [90, 90],
        [-40, -40],
        [66.69405999290854, 66.69405999290854],
        [-158.62725034, -158.62725034],
        [157.03891041, 157.03891041],
        [-56.08440611, -56.08440611],
    ],
)
coefficients_3 = minimize(
    sum_residuals_chi,
    [0, 0, 0, 0, 0, 0],
    args=(mass_full, period_full, 0.46),
    bounds=[
        [90, 90],
        [-45, -45],
        [66.69405999290854, 66.69405999290854],
        [-158.62725034, -158.62725034],
        [157.03891041, 157.03891041],
        [-56.08440611, -56.08440611],
    ],
)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.set(
    xlim=(1.5, 0), ylim=(-1, 35), xlabel=r"Mass ($M_\odot$)", ylabel="Period ($days$)"
)
ax.scatter(mass_full, period_full, color="#189ad3")
ax.plot(
    white_space,
    general_polynomial_chi(coefficients_3.x, white_space, 0.86),
    linewidth=2.2,
    c="#1BE7FF",
)
ax.plot(
    white_space,
    general_polynomial_chi(coefficients_2.x, white_space, 0.86),
    linewidth=2.2,
    c="#FFBC42",
)
ax.plot(
    white_space,
    general_polynomial_chi(coefficients_1.x, white_space, 0.86),
    linewidth=2.2,
    c="#E15A45",
)

ax.text(1.10, 11, "SLOW", rotation=23, color="#4C4C4C")

ax.text(0.21, 3, "FAST", rotation=0, color="#4C4C4C")

ax.text(0.35, 7, "TRANSITION", rotation=277, color="#4C4C4C")

fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/chiswitch_model.png",
    dpi=800,
    bbox_inches="tight",
    # transparent=True,
    # facecolor="#E5E5E5",
)
#%% FITTED PLOT
coefficients = minimize(
    sum_residuals_chi,
    [0, 0, 0, 0, 0, 0],
    args=(mass_full, period_full, 0.46),
    bounds=[
        [90, 90],
        [-40, -40],
        [66.69405999290854, 66.69405999290854],
        [-158.62725034, -158.62725034],
        [157.03891041, 157.03891041],
        [-56.08440611, -56.08440611],
    ],
)

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.set(
    xlim=(1.5, 0), ylim=(-1, 35), xlabel=r"Mass ($M_\odot$)", ylabel="Period ($days$)"
)
# Data and line
ax.scatter(mass_full, period_full, color="#189ad3")
ax.plot(
    white_space,
    general_polynomial_chi(coefficients.x, white_space, 0.86),
    linewidth=1.4,
    c="#E15A45",
)

start, split, end = -3, 3, 35
ax.plot(
    gaussian(np.linspace(start, split, 100), 1, 0.5) * 0.02 + 1,
    np.linspace(start, split, 100),
    c="#FFBC42",
    linewidth=4,
)
ax.plot(
    gaussian(np.linspace(split, end, 100), 9, 1.5) * 0.09 + 1,
    np.linspace(split, end, 100),
    c="#FFBC42",
    linewidth=4,
)

start, split, end = -3, 12, 35
ax.plot(
    gaussian(np.linspace(start, split, 100), 0.86, 2.4) * 0.05 + 0.44,
    np.linspace(start, split, 100),
    c="#FFBC42",
    linewidth=4,
    label="Flattened Probability\nDistribution of P ",
)
ax.plot(
    gaussian(np.linspace(split, end, 100), 20, 3) * 0.12 + 0.44,
    np.linspace(split, end, 100),
    c="#FFBC42",
    linewidth=4,
)

vertical_white_space = np.linspace(-3, 35, 100)
ax.plot(
    gaussian(vertical_white_space, 0.86, 1.5) * 0.09 + 0.23,
    vertical_white_space,
    c="#FFBC42",
    linewidth=4,
)

ax.text(0.75, 6, "Non-Zero", color="#4C4C4C", ha="center", va="center")
ax.annotate(
    "",
    (0.44, 10),
    (0.63, 6),
    arrowprops={"arrowstyle": "->", "color": "#4C4C4C", "lw": 2.5},
    va="center",
    fontsize="25",
    color="#4C4C4C",
)
ax.annotate(
    "",
    (1, 4),
    (0.86, 6),
    arrowprops={"arrowstyle": "->", "color": "#4C4C4C", "lw": 2.5},
    va="center",
    fontsize="25",
    color="#4C4C4C",
)

ax.legend()
fig.savefig(
    "C:/Users/elydo/Documents/Harvard/AAS Poster stuff/period_probability.png",
    dpi=800,
    bbox_inches="tight",
)
