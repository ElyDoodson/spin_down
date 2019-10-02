#%% MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%% FUNCTIONS


def app_to_abs(app_mag, distance_parsec):
    return app_mag - (5 * np.log10(distance_parsec)) + 5


#%%

path = "D:/dev/spin_down/new_data/"

cluster_list = os.listdir(path)
files = [os.listdir(path + str(cluster)) for cluster in cluster_list]


# dict_list = [{"name": cluster, "data_frame": np.concatenate([pd.read_csv(path + file) for file in files])} for index, cluster in enumerate(cluster_list)]

[file for file in files]

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

[datum["data_frames"] for datum in dict_list]