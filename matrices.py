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
            self.features = np.array([1,mass,mass**2])


def get_data(location):
    """
    Takes string as input and returns a list of Star objects
    """
    data_frame = pd.read_csv(location, encoding="utf-8", delimiter="\t", comment="#")

    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()

    # return period, np.array([np.array([1,m,m**2]) for m in mass ])
    return [Star(period, mass) for mass, period in zip(mass, period)]

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

    return float(np.dot(star_attributes, line_data))

#%% DATA INITIALISATION
path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

period_list, predictor_list = get_data(path)
group_value = 
line_slow = []
#%% TEST CELL

line_data = [-1.22,1,0]
star = Star(1.2, 0.4).features
predict_value(star,line_data)