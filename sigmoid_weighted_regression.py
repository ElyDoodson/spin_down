#%% IMPORTING MODUKES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random

#%%  INITIALISATION OF FUNCTIONS


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = mass


def get_data(location):
    data_frame = pd.read_csv(location, encoding="utf-8", delimiter="\t", comment="#")
    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()

    return [Star(period, mass) for mass, period in zip(mass, period)]


def set_initial_star_group(star_list, order):
    if order < 2:
        coefficients = np.array([5, -5])
    if order >= 2:
        coefficients = np.append([5, -5], np.zeros(order - 2))
    for star in star_list:
        star.group = (
            1 if predict_value(star.predictors, coefficients) <= star.period else 0
        )


def set_predictor_order(star_list, order):
    for star in star_list:
        star.predictors = [star.mass ** i for i in range(order)]


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
        dist_slow = (slow_linreg.predict([star.predictors]) - star.period) ** 2
        dist_fast = (fast_linreg.predict([star.predictors]) - star.period) ** 2

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


def calculate_coefficients(star_objects, group, sample_weights=None, set_slope=False):
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

    if set_slope == True:
        # sets the coefs to zero
        lr.coef_ = np.zeros(len(star_objects[0].predictors))
        lr.intercept_ = np.average(
            [star.period for star in star_list if star.group == group],
            weights=sample_weights,
        )

    return np.append(lr.intercept_, lr.coef_[1:]), lr


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
    if selected_group != 0 or selected_group != 1:

        star_periods = np.array([star.period for star in star_list])
        star_predictors = [star.predictors for star in star_list]

    if selected_group == 0 or selected_group == 1:
        # Calculating recuring values
        star_periods = np.array(
            [star.period for star in star_list if star.group == selected_group]
        )
        star_predictors = [
            star.predictors for star in star_list if star.group == selected_group
        ]

    star_predicted_fast = lr_fast.predict(star_predictors)
    star_predicted_slow = lr_slow.predict(star_predictors)

    chi = np.divide(
        (star_periods - star_predicted_slow) + (star_periods - star_predicted_fast),
        (star_predicted_slow - star_predicted_fast),
    )
    weight_slow = sigmoid(chi)  # - sigmoid(chi, steepness= 3, offest= - 7)
    if selected_weight == "slow":
        return weight_slow
    elif selected_weight == "fast":
        return 1 - weight_slow


def sigmoid(x, steepness=16, offest=10):
    """
    Takes any range of input and returns a value between 0 and 1 with the distribution 
    shown at https://www.desmos.com/calculator/r84xmc80id
    """
    return np.divide(1, 1 + np.exp(-(steepness * x + offest)))


def calculate_chi(star_list, lr_slow, lr_fast):
    star_period = [star.period for star in star_list]
    star_predictors = [star.predictors for star in star_list]

    star_predict_slow = lr_slow.predict(star_predictors)
    star_predict_fast = lr_fast.predict(star_predictors)

    return np.divide(
        (star_period - star_predict_slow) + (star_period - star_predict_fast),
        (star_predict_slow - star_predict_fast),
    )


# def plot_lines(axes, lrs, lrf, x=line_mass, x_x=line_predictors):
#     axes.plot(x, lrs.predict(x_x), color="blue")
#     axes.plot(x, lrf.predict(x_x), color="red")
#     return axes


#%% DATA INITIALISATION

path = "d:data\Praesepe_K2.csv"
path = "D:\dev\spin_down\data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"
object_list = get_data(path)

X_train, X_test, y_train, y_test = train_test_split(
    [star.mass for star in object_list],
    [star.period for star in object_list],
    test_size=0.2,
    random_state=42,
)

star_list = [Star(y, x) for x, y in zip(X_train, y_train)]
star_list_test = [Star(y, x) for x, y in zip(X_test, y_test)]
#%%
order = 3
set_predictor_order(star_list, order)
set_initial_star_group(star_list, order)

line_predictors = [[q ** i for i in range(order)] for q in np.linspace(1.4, 0.2, 100)]
line_mass = [item[1] for item in line_predictors]

#%% INITIAL FIT
coefficients_slow, lrs = calculate_coefficients([star for star in star_list], group=1)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list], group=0, set_slope=True
)


fig, ax = plt.subplots(1, figsize=(9, 6))
ax.set(ylim=(-1, 15), xlim=(1.6, 0), title="GROUP SPLIT BY LINE 5X + 7")

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
ax.plot(line_mass, lrf.predict(line_predictors), color="red")
ax.plot(line_mass, lrs.predict(line_predictors), color="blue")

print("Slow coeff =", coefficients_slow)
print("Fast coeff =", coefficients_fast)


#%% SET GROUPS TO CLOSEST LINE

set_star_group(star_list, lrs, lrf)

coefficients_slow, lrs = calculate_coefficients([star for star in star_list], group=1)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list], group=0, set_slope=True
)

fig, ax = plt.subplots(1, figsize=(9, 6))
ax.set(ylim=(-1, 15), xlim=(1.6, 0), title="GROUPS TO CLOSEST LINE")

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
ax.plot(line_mass, lrf.predict(line_predictors), color="red")
ax.plot(line_mass, lrs.predict(line_predictors), color="blue")

print("Slow coeff =", coefficients_slow)
print("Fast coeff =", coefficients_fast)


#%% USE WEIGHTED FITTING
set_star_group(star_list, lrs, lrf)

sample_weight_slow = calculate_weight(star_list, lrs, lrf, 1, "slow")
sample_weight_fast = calculate_weight(star_list, lrs, lrf, 0, "fast")

coefficients_slow, lrs = calculate_coefficients(
    [star for star in star_list], group=1, sample_weights=sample_weight_slow
)
coefficients_fast, lrf = calculate_coefficients(
    [star for star in star_list],
    group=0,
    set_slope=True,
    sample_weights=sample_weight_fast,
)

fig, ax = plt.subplots(1, figsize=(9, 6))
ax.set(ylim=(-1, 15), xlim=(1.6, 0), title="WEIGHTED FITTING")

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
ax.plot(line_mass, lrf.predict(line_predictors), color="red")
ax.plot(line_mass, lrs.predict(line_predictors), color="blue")

print("slow coef = ", coefficients_slow)
print("fast coef = ", coefficients_fast)

#%% RANDOM STAR ATTRIBUTE CHECKER

nmbr = random.randint(1, len(star_list))

fig, ax = plt.subplots(1, figsize=(9, 6))
ax.set(ylim=(-1, 15), xlim=(1.6, 0), title="Visualisation of Weights")

ax.scatter(
    [star.mass for star in star_list],
    [star.period for star in star_list],
    marker="x",
    c=calculate_weight(star_list, lrs, lrf, "both", "fast"),
    cmap="coolwarm",
)
ax.plot(line_mass, lrf.predict(line_predictors), color="red")
ax.plot(line_mass, lrs.predict(line_predictors), color="blue")

ax.scatter(star_list[nmbr].mass, star_list[nmbr].period, marker="o", color="green")
ax.scatter(
    [star.mass for star in star_list_test],
    [star.period for star in star_list_test],
    marker="o",
    color="orange",
    s=10,
)
print(
    "\n",
    "slow_weight of star selected = ",
    calculate_weight(star_list, lrs, lrf, "both groups", "slow")[nmbr],
    "\n",
    "group = ",
    star_list[nmbr].group,
)

#%%
lrs.predict([star_list[0].predictors])

#%%
fig, ax2 = plt.subplots(1, figsize=(9, 6))
ax2.set(
    title="Visualisation of Weights",
    xlabel="Polynomial Order",
    ylabel="Sum of Residuals",
)

for order in range(2, 20, 1):
    set_predictor_order(star_list, order)
    set_initial_star_group(star_list, order)

    line_predictors = [
        [q ** i for i in range(order)] for q in np.linspace(1.4, 0.2, 100)
    ]
    line_mass = [item[1] for item in line_predictors]

    coefficients_slow, lrs = calculate_coefficients(
        [star for star in star_list], group=1
    )
    coefficients_fast, lrf = calculate_coefficients(
        [star for star in star_list], group=0, set_slope=True
    )

    for i in range(30):
        set_star_group(star_list, lrs, lrf)

        sample_weight_slow = calculate_weight(star_list, lrs, lrf, 1, "slow")
        sample_weight_fast = calculate_weight(star_list, lrs, lrf, 0, "fast")

        coefficients_slow, lrs = calculate_coefficients(
            [star for star in star_list], group=1, sample_weights=sample_weight_slow
        )
        coefficients_fast, lrf = calculate_coefficients(
            [star for star in star_list],
            group=0,
            set_slope=True,
            sample_weights=sample_weight_fast,
        )
    # fig, ax = plt.subplots(1, figsize=(9, 6))
    # ax.set(ylim=(-1, 15), xlim=(1.6, 0), title="Visualisation of Weights")

    # ax.scatter(
    #     [star.mass for star in star_list],
    #     [star.period for star in star_list],
    #     marker="x",
    #     c=calculate_weight(star_list, lrs, lrf, "both", "fast"),
    #     cmap="coolwarm",
    # )
    # ax.plot(line_mass, lrf.predict(line_predictors), color="red")
    # ax.plot(line_mass, lrs.predict(line_predictors), color="blue")

    set_predictor_order(star_list_test, order)
    set_star_group(star_list_test, lrs, lrf)
    L = np.sum(
        [
            (star.period - lrs.predict([star.predictors])) ** 2
            for star in star_list_test
            if star.group == 1
        ]
    )

    ax2.scatter(order, L, color="white")
    print(order)
#%%
