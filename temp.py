#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from astropy.table import Table
from astropy.io import fits


filename1 = "D:\dev\spin_down/new_MIST_data/mist_with_tau/00010M.track.eep"

column_names = [
    "star_age",
    "star_mass",
    "star_mdot",
    "he_core_mass",
    "c_core_mass",
    "o_core_mass",
    "log_L",
    "log_L_div_Ledd",
    "log_LH",
    "log_LHe",
    "log_LZ",
    "log_Teff",
    "log_abs_Lgrav",
    "log_R",
    "log_g",
    "log_surf_cell_z",
    "surf_avg_omega",
    "surf_avg_v_rot",
    "surf_num_c12_div_num_o16",
    "v_wind_Km_per_s",
    "surf_avg_omega_crit",
    "surf_avg_omega_div_omega_crit",
    "surf_avg_v_crit",
    "surf_avg_v_div_v_crit",
    "surf_avg_Lrad_div_Ledd",
    "v_div_csound_surf",
    "surf_r_equatorial_div_r",
    "surf_r_polar_div_r",
    "total_angular_momentum",
    "surface_h1",
    "surface_he3",
    "surface_he4",
    "surface_li7",
    "surface_be9",
    "surface_b11",
    "surface_c12",
    "surface_c13",
    "surface_n14",
    "surface_o16",
    "surface_f19",
    "surface_ne20",
    "surface_na23",
    "surface_mg24",
    "surface_al26",
    "surface_si28",
    "surface_s32",
    "surface_ca40",
    "surface_ti48",
    "surface_fe56",
    "log_center_T",
    "log_center_Rho",
    "center_degeneracy",
    "center_omega",
    "center_gamma",
    "mass_conv_core",
    "center_h1",
    "center_he4",
    "center_c12",
    "center_n14",
    "center_o16",
    "center_ne20",
    "center_mg24",
    "center_si28",
    "pp",
    "cno",
    "tri_alfa",
    "burn_c",
    "burn_n",
    "burn_o",
    "c12_c12",
    "delta_nu",
    "delta_Pg",
    "nu_max",
    "acoustic_cutoff",
    "conv_env_top_mass",
    "conv_env_bot_mass",
    "conv_env_top_radius",
    "conv_env_bot_radius",
    "conv_env_turnover_time_l_t",
    "conv_env_turnover_time_l_b",
    "conv_env_turnover_time_g",
    "max_conv_vel_div_csound",
    "max_gradT_div_grada",
    "gradT_excess_alpha",
    "min_Pgas_div_P",
    "max_L_rad_div_Ledd",
    "e_thermal",
    "envelope_binding_energy",
    "k_2",
    "moment_of_inertia",
    "phase",
]
# df = pd.read_csv(
#     filename1, comment="#", sep=" ", names=column_names, skipinitialspace=True
# )

path = "D:\dev\spin_down/new_MIST_data/mist_with_tau/"
files = os.listdir(path)

df = pd.concat(
    [
        pd.read_csv(
            path + file, comment="#", sep=" ", names=column_names, skipinitialspace=True
        )
        for file in files
    ]
)

df.to_csv("D:\dev\spin_down/new_MIST_data/tau_mist.csv", sep="\t", index=False)
