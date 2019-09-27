#%% MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% FUNCTIONS


def abs_from_app(vmag, distance_parsec):
    return vmag - (5 * np.log10(distance_parsec)) + 5


def vmag_to_mass(vmag):
    c0 = 0.19226
    c1 = -0.050737
    c2 = 0.010137
    c3 = -0.00075399
    c4 = -1.9858e-05
    x0 = 13.0

    first = c1 * (vmag - x0)
    second = c2 * (vmag - x0) ** 2
    third = c3 * (vmag - x0) ** 3
    forth = c4 * (vmag - x0) ** 4
    return c0 + first + second + third + forth


#%%
path = "D:\dev\spin_down\OCdata/"
source = "M37PC.tsv"
dist = 1383
age = 550


df = pd.read_csv(path + source, sep="\t", comment="#")


periods = df.Prot.to_numpy()
vmag = df.Vmag.to_numpy()


print(df.tail())
print(df.Vmag.describe())

abs_vmag = abs_from_app(vmag, dist)

masses = vmag_to_mass(abs_vmag)

converted_df = pd.DataFrame({"Per": periods, "M": masses})


converted_df.to_csv(
    path + "p_m/" + source[:3] + "pm_" + str(age) + ".csv",
    sep="\t",
    index=None,
    header=True,
)
#%%

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.invert_xaxis()
ax.set(ylim=(-2, 30), title=source[:-4])
ax.scatter(masses, periods)
