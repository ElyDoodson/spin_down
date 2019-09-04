import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
from os import listdir
import pandas as pd


# Don't forget the trailing '/'
source = "data/in/here/"
data_frames = [
    pd.read_csv(source + file_name, encoding="utf-8", delimiter="\t")
    for file_name in listdir(source)
    if "~" in file_name
]
"""
df1 = pd.read_csv("d:data\Praesepe_K2.csv", encoding="utf-8", delimiter="\t")
df2 = pd.read_csv("d:data\hPer.csv", encoding="utf-8", delimiter="\t")
df3 = pd.read_csv("d:data\Hyades_K2.csv", encoding="utf-8", delimiter="\t")
df4 = pd.read_csv(
    "d:data\Pleiades_Hartman.csv", encoding="utf-8", delimiter="\t", skiprows=1
)
df5 = pd.read_csv("d:data\M37_Hartman.csv", encoding="utf-8", delimiter="\t")

fig, ax = plt.subplots(2, 3, sharex="col", sharey="row", figsize=(6, 3))
for datum in [df1, df2, df3, df4]:

    mass = datum.M.tolist()
    period = datum.Per.tolist()
    for i in range(5):
        r, c = divmod(i, 3)
        print(r, c)
        ax[r][c].plot(mass, period, linestyle="none", marker="x")


ax[1][0].invert_xaxis()
plt.show()

# mass = df1.M.tolist()
# period = df1.Per.tolist()

# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize = (6,3))
# ax[0][0].plot(mass, period, linestyle = "none", marker = "x")
# ax[1][0].invert_xaxis()

# df.head()
