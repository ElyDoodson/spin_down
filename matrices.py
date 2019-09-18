#%% IMPORTING MODUKES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

# from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

#%%  INITIALISATION OF FUNCTIONS


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = mass
        self.predictors = np.array([1, mass, mass ** 2])


def get_data(location):
    """
    Takes string as input and returns a list of Star objects
    """
    data_frame = pd.read_csv(location, encoding="utf-8", delimiter="\t", comment="#")

    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()

    return [Star(period, mass) for mass, period in zip(mass, period)]


def set_initial_star_group(star_list, coefficients=[7, -5, 0]):
    """
    
    """
    for star in star_list:
        star.group = (
            1 if predict_value(star.predictors, coefficients) <= star.period else 0
        )


def predict_value(star_predictors, line_data):
    """
    Parameters
    ------
        star_predictors: array-like of M by n dimensions
            dimensions must match line_data dimensions
            [1, mass, mass**2, ...], 

        line_data: array-like of 1 by n dimensions
            contains coefficients of like [b0, b1, b2, ...]

    Returns
    -------
    y_hat: int, dot product of the matrices. i.e. the predicted value
    """

    return np.dot(star_predictors, line_data)


def calculate_coefficients(
    star_objects, group, sample_weights=1, return_linregg=False, set_slope=False
):
    """
    Parameters
    --------
        star_objects: list
            Contains a list of Star objects
        group: int, 1 or 0
            1 is slow rotators, 0 is fast
        sample_weights: list, optional, default 1
            containts list of weights of the values w.r.t. the line
        return_linregg: boolean, optional, default False
            Whether to return the object of the linear regression to 
            use things such as predict
        set_slope: boolean, optional, default False
            Sets the slope to 0 
    Returns
    ------
    LinearRegression: class object, optional, default False
    list: [b0,b1,b2] 
    """
    lr = LinearRegression()

    lr.fit(
        [star.predictors for star in star_objects if star.group == group],
        [star.period for star in star_list if star.group == group],
        sample_weights,
    )

    lr.coef_ = lr.coef_ if set_slope == False else np.zeros(len(lr.coef_))

    if return_linregg == True:
        return lr.intercept_ + lr.coef_[1:], lr

    elif return_linregg == False:
        return lr.intercept_ + lr.coef_[1:]

    else:
        print("return_linregg must be a boolean")
        return None


#%% DATA INITIALISATION
path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

star_list = get_data(path)
set_initial_star_group(star_list)


#%%
coefficients_slow, lrs = calculate_coefficients(
    [star for star in star_list], group=0, return_linregg=True, set_slope=True
)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list], group=1, return_linregg=True
)


fig, ax = plt.subplots(1)
ax.invert_xaxis()
ax.scatter(
    [star.mass for star in star_list if star.group == 1],
    [star.period for star in star_list if star.group == 1],
    marker="x",
    color="blue",
)
ax.scatter(
    [star.mass for star in star_list if star.group == 0],
    [star.period for star in star_list if star.group == 0],
    marker="x",
    color="red",
)
ax.scatter(
    [star.mass for star in star_list],
    lrs.predict([star.predictors for star in star_list]),
    color="red",
)
ax.scatter(
    [star.mass for star in star_list],
    lrf.predict([star.predictors for star in star_list]),
    color="blue",
)


#%%
