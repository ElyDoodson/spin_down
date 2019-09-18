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


def set_star_group(star_list, slow_linreg, fast_linreg):
    """
    Sets the group of the stars in star_list to the closest line

    Parameters
    -------
        star_list: list
            list of all star objects
        slow_linreg: LinearRegression() obj
            object correspoding to the slow regression line
        fast_linreg: LinearRegression() obj
            object correspoding to the fast regression line
    Returns
    ------
        None

    """
    for star in star_list:
        dist_slow = (slow_linreg.predict([[1, star.mass, star.mass ** 2]]) - star.period)**2
        dist_fast = (fast_linreg.predict([[1, star.mass, star.mass ** 2]]) - star.period)**2

        # dist_slow = (
        #     predict_value(star.predictors, slow_coefficients) - star.period
        # ) ** 2
        # dist_fast = (
        #     predict_value(star.predictors, fast_coefficients) - star.period
        # ) ** 2
        star.group = 1 if min(dist_fast, dist_slow) == dist_slow else 0


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
        return np.append(lr.intercept_, lr.coef_[1:]), lr

    elif return_linregg == False:
        return np.append(lr.intercept_, lr.coef_[1:])

    else:
        print("return_linregg must be a boolean")
        return None


# def calculate_ser(predict_y1, predict_y0, y):
#     return (predict_y1 - y) ** 2 / (predict_y0 - y) ** 2


def calculate_weight(star_list, lr_slow, lr_fast, selected_group, selected_weight):
    """
    Takes the star list and calculates the weight of the stars w.r.t 
    the chosen line

    Parameters
    -------
        star_list: list
            List of Star objects
        coef_slow: list
            list of slow coefficients
        coef_fast: list
            list of fast coefficients

        selected_group: int
            Selected group of stars you wish to work with. 1 or 0 (slow or fast)
        selected_weight: str, "slow" or "fast"
            Selected weight to return
    
    Returns
    -------
        list: list of weights of the selected star group w.r.t the selected line
    """
    # Calculating recuring values
    star_periods = np.array(
        [star.period for star in star_list if star.group == selected_group]
    )
    star_predictors = [
        star.predictors for star in star_list if star.group == selected_group
    ]

    # Calculating square of the residuals
    square_res = np.divide(
        np.square(lr_slow.predict(star_predictors) - star_periods),
        np.square(lr_fast.predict(star_predictors) - star_periods),
    )

    weight_slow = np.divide(1.0, (1 + square_res))
    # weight_fast = np.divide(1., (1 + np.divide(1,square_res)))
    if selected_weight == "slow":
        return weight_slow
    elif selected_weight == "fast":
        return 1 - weight_slow


#%% DATA INITIALISATION
path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

star_list = get_data(path)
set_initial_star_group(star_list)


#%%
coefficients_slow, lrs = calculate_coefficients(
    [star for star in star_list], group=1, return_linregg=True
)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list], group=0, return_linregg=True, set_slope=True
)


fig, ax = plt.subplots(1, figsize=(8, 6))
ax.invert_xaxis()
ax.set(title="GROUP SPLIT BY LINE 5X + 7")
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
ax.plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrs.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="blue",
)
ax.plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrf.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="red",
)
print("Slow coeff =", coefficients_slow)
print("Fast coeff =", coefficients_fast)


#%% SET GROUPS TO CLOSEST LINE

set_star_group(star_list, lrs, lrf)

coefficients_slow, lrs = calculate_coefficients(
    [star for star in star_list], group=1, return_linregg=True
)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list], group=0, return_linregg=True, set_slope=True
)

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.invert_xaxis()
ax.set(title="GROUPS TO CLOSEST LINE")
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
ax.plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrs.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="blue",
)
ax.plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrf.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="red",
)
print("Slow coeff =", coefficients_slow)
print("Fast coeff =", coefficients_fast)


#%%
set_star_group(star_list, lrs, lrf)
sample_weight_slow = calculate_weight(
    star_list, lrs, lrf, 1, "slow"
)
sample_weight_fast = calculate_weight(
    star_list, lrs, lrf, 0, "fast"
)

coefficients_slow, lrs = calculate_coefficients(
    [star for star in star_list],
    group=1,
    return_linregg=True,
    sample_weights=sample_weight_slow,
)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list],
    group=0,
    return_linregg=True,
    set_slope=True,
    sample_weights=sample_weight_fast,
)

fig, ax = plt.subplots(2, figsize=(8, 6))
ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[0].set(title="NaN")
ax[0].scatter(
    [star.mass for star in star_list if star.group == 1],
    [star.period for star in star_list if star.group == 1],
    marker="x",
    color="blue",
)
ax[0].scatter(
    [star.mass for star in star_list if star.group == 0],
    [star.period for star in star_list if star.group == 0],
    marker="x",
    color="red",
)
ax[0].plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrs.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="blue",
)
ax[0].plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    lrf.predict([[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)]),
    color="red",
)

ax[1].scatter(
    [star.mass for star in star_list if star.group == 1],
    [star.period for star in star_list if star.group == 1],
    marker="x",
    color="blue",
)
ax[1].scatter(
    [star.mass for star in star_list if star.group == 0],
    [star.period for star in star_list if star.group == 0],
    marker="x",
    color="red",
)
ax[1].plot(
    [i for i in np.arange(1.4, 0.2, -0.01)],
    predict_value(
        [[1, i, i ** 2] for i in np.arange(1.4, 0.2, -0.01)], coefficients_slow
    ),
    color="blue",
)


print("Slow coeff =", coefficients_slow)
print("Fast coeff =", coefficients_fast)


#%%
