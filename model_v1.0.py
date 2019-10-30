#%%
import numpy as np

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
path = "D:/dev/spin_down/mistmade_data/praesepe_pm.csv"
df = pd.read_csv(path, sep="\t")

fig, ax = plt.subplots(1, figsize=(11.5, 7))
ax.invert_xaxis()
ax.scatter(df.Mass.to_numpy(), df.Per.to_numpy(), label="Raw Data", c="red")


coeffs = minimize(mse, [0, 0, 0, 0, 0], args=(df.Mass.to_numpy(), df.Per.to_numpy()))

z = np.linspace(2,0, 200)
ax.scatter(
    z,
    y_pred_test(coeffs.x, z),
    label="Pred from minimised coeffs",
    marker="x",
    c="green",
)
ax.legend()


# %%
