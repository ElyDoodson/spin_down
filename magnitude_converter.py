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
