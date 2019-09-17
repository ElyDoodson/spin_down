#%% IMPORTING MODUKES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

#%%  INITIALISATION OF FUNCTIONS


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.predictors = np.array([1, mass, mass ** 2])


def get_data(location):
    """
    Takes string as input and returns a list of Star objects
    """
    data_frame = pd.read_csv(location, encoding="utf-8", delimiter="\t", comment="#")

    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()

    # return period, np.array([np.array([1,m,m**2]) for m in mass ])
    return [Star(period, mass) for mass, period in zip(mass, period)]


def set_initial_star_group(star_list, predictors=[7, -5, 0]):
    """
    
    """
    for star in star_list:
        star.group = (
            1 if predict_value(star.predictors, seperation_line) >= star.period else 0
        )


def predict_value(star_attributes, line_data):
    """
    Parameters
    ------
        star_attributes: array-like of n dimensions
            dimensions must match line_data dimensions
            [1, mass, mass**2, ...], 
        line_data: array-like of n dimensions
            contains coefficients of like [b0, b1, b2, ...]
    Returns
    -------
    y_hat: int, dot product of the matrices. i.e. the predicted value
    """

    return np.dot(star_attributes, line_data)


#%% DATA INITIALISATION
path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

star_list = get_data(path)
set_initial_star_group(star_list)


#%% TEST CELL


ad
