#%%
import numpy as np
import os

# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics.scorer import make_scorer
# from sklearn.svm import SVR
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

#%%
def y_pred(m, b0, b2, b3, b4, b5):
    return b0 + (b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5))))


def y_pred_test(params, m):
    b0, b2, b3, b4, b5 = params
    return b0 + (b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5))))


def mse(params, m, period):
    b0, b2, b3, b4, b5 = params

    return np.sum(
        (b0 + (b2 * m + b3) * (1 / (1 + np.exp(-(b4 * m + b5)))) - period) ** 2
    ) / len(period)


#%% DATA IMPORTATION
# path = "D:/dev/spin_down/mistmade_data/praesepe_pm"
# df = pd.read_csv(path, sep="\t")

# fig, ax = plt.subplots(1, figsize=(11.5, 7))
# ax.invert_xaxis()
# ax.scatter(df.Mass.to_numpy(), df.Per.to_numpy(), label="Raw Data", c="red")


# coeffs = minimize(mse, [0, 0, 0, 0, 0], args=(df.Mass.to_numpy(), df.Per.to_numpy()))

# z = np.linspace(2,0, 200)
# ax.scatter(
#     z,
#     y_pred_test(coeffs.x, z),
#     label="Pred from minimised coeffs",
#     marker="x",
#     c="green",
# )
# ax.legend()


#%% Cluster Dictionary
path = "D:/dev/spin_down/mistmade_data/"
files = os.listdir(path)
cluster_list = [name[:-4] for name in files]
cluster_dict = {}
names = np.array([name[:-16] for name in cluster_list])

# Makes dictionary with namee access to cluster, containing "age" and "df"
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
#Deleting the stand-alone clusters
del cluster_dict["praesepe"]
del cluster_dict["m50"]
del cluster_dict["m35"]
del cluster_dict["ngc2516"]
del cluster_dict["m34"]
del cluster_dict["ngc2301"]

#%%
#redeclaring names after removal
names = np.array(list(cluster_dict.keys()))

fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(24, 16))
fig.subplots_adjust(wspace=0.03, hspace=0.05)
fig2, ax2 = plt.subplots(1, figsize=(11.5, 7))


sorted_names = names[np.argsort([cluster_dict[name]["age"] for name in names])]

for num, name in enumerate(sorted_names):
    r, c = divmod(num, 3)
    # print(r, c)

    mass = cluster_dict[name]["df"]["Mass"]
    period = cluster_dict[name]["df"]["Per"]

    ax[r][c].invert_xaxis()

    ax[r][c].plot(
        mass,
        period,
        linestyle="none",
        marker="x",
        label=("%s %iMYrs") % (name, cluster_dict[name]["age"]),
        markersize=1.5,
    )

    coeffs = minimize(mse, [0, 0, 0, 0, 0], args=(mass, period))

    z = np.linspace(2, 0, 200)
    ax[r][c].scatter(
        z,
        y_pred_test(coeffs.x, z),
        label="Pred from minimised coeffs",
        marker="x",
        c="green",
    )
    ax[r][c].legend()
    ax[r][c].set(xlim=(1.4, 0.0), ylim=(-2, 20.0))  # ylim = (0, 100), yscale = "log",)

    ax2.scatter(
        [cluster_dict[name]["age"]] * len(coeffs.x),
        coeffs.x,
        label=cluster_dict[name]["age"],
    )
    ax2.legend()
    ax2.set(ylim=(-100, 100))
fig.show()
fig2.show()

#%%
