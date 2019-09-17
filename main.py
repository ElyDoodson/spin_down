#%% MODULE IMPORTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

#%% FUNCTION INITIALISATION
class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = mass

    # def set_initial_group(self):
    #     self.group = 1 if self.period >= calculate_line(-5, self.mass, 7) else 0

    def update_group(self, line_list):
        # dist_slow = (line_list[1].predicted_value(self.mass) - self.period) ** 2
        # dist_fast = (line_list[0].predicted_value(self.mass) - self.period) ** 2
        dist_slow = (self.predict_slow(line_list) - self.period) ** 2
        dist_fast = (self.predict_fast(line_list) - self.period) ** 2
        self.group = 1 if min(dist_fast, dist_slow) == dist_slow else 0

    def calculate_weight_s(self, line_list):
        # return 1 / (
        #     1
        #     + calculate_ser(
        #         line_list[1].predicted_value(self.mass),
        #         line_list[0].predicted_value(self.mass),
        #         self.period,
        #     )
        # )
        return 1 / (
            1
            + calculate_ser(
                self.predict_slow(line_list), self.predict_fast(line_list), self.period
            )
        )

    def calculate_weight_f(self, line_list):
        # return 1 / (
        #     1
        #     + (
        #         1
        #         / calculate_ser(
        #             line_list[1].predicted_value(self.mass),
        #             line_list[0].predicted_value(self.mass),
        #             self.period,
        #         )
        #     )
        # )
        return 1 / (
            1
            + (
                1
                / calculate_ser(
                    self.predict_slow(line_list),
                    self.predict_fast(line_list),
                    self.period,
                )
            )
        )

    def predict_slow(self, line_list):
        return calculate_line(line_list[1][0], self.mass, line_list[1][1])

    def predict_fast(self, line_list):
        return calculate_line(line_list[0][0], self.mass, line_list[0][1])


# class Line:
#     def __init__(self, slope, intercept):
#         self.slope = slope
#         self.intercept = intercept

#     def updatefor_slowstars(self, star_list):
#         """
#         this updates the slope of a line for the true slope and intercept
#         """
#         lrs = LinearRegression()
#         lrs.fit(
#             [[star.mass] for star in star_list if star.group == 1],
#             [star.period for star in star_list if star.group == 1],
#             sample_weight=[
#                 star.calculate_weight_s(line_list)
#                 for star in star_list
#                 if star.group == 1
#             ],
#         )
#         self.slope = lrs.coef_
#         self.intercept = lrs.intercept_
#         # slope_s, intercept_s = linregress(
#         #     [star.mass for star in star_list if star.group == 1],
#         #     [star.period for star in star_list if star.group == 1],
#         # )[:2]
#         # self.slope = slope_s
#         # self.intercept = intercept_s

#     def updatefor_faststars(self, star_list):
#         """
#         This updates the slope of the line using the average of the data.
#         Effectively fixing the slope at 0 and only changing its intercept
#         """
#         lrf = LinearRegression()
#         lrf.fit(
#             [[star.mass] for star in star_list if star.group == 0],
#             [star.period for star in star_list if star.group == 0],
#             sample_weight=[
#                 star.calculate_weight_f(line_list)
#                 for star in star_list
#                 if star.group == 0
#             ],
#         )
#         self.slope = np.array([0])
#         self.intercept = lrf.intercept_

#         # slope_f, intercept_f = (
#         #     0,
#         #     np.sum([star.period for star in star_list if star.group == 0])
#         #     / len([star.group for star in star_list if star.group == 0]),
#         # )
#         # self.slope = slope_f
#         # self.intercept = intercept_f

#     def predicted_value(self, star):
#         return calculate_line(self.slope, star, self.intercept)


def calculate_ser(predict_y1, predict_y0, y):
    return (predict_y1 - y) ** 2 / (predict_y0 - y) ** 2


def calculate_line(m, x, c):
    return np.add(np.dot(m, x), c)


def get_data(location):
    """
    Takes string as input and returns a list of Star objects
    """
    data_frame = pd.read_csv(location, encoding="utf-8", delimiter="\t", comment="#")

    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()

    return [Star(period, mass) for mass, period in zip(mass, period)]


# def get_lines(star_list):
#     """
#     Takes in list of star objects and finds the initial lines
#     """
#     slope_s, intercept_s = linregress(
#         [star.mass for star in star_list if star.group == 1],
#         [star.period for star in star_list if star.group == 1],
#     )[:2]
#     slope_f, intercept_f = (
#         0,
#         np.sum([star.period for star in star_list if star.group == 0])
#         / len([star.group for star in star_list if star.group == 0]),
#     )

#     return [Line(slope_f, intercept_f), Line(slope_s, intercept_s)]


def plot_data(star_list, line_list, figure="figure", axes="ax"):
    figure, ax = plt.subplots(1, figsize=(6, 4))
    ax.invert_xaxis()
    ax.set(ylim=(0, 15))
    ax.scatter(
        [star.mass for star in star_list if star.group == 0],
        [star.period for star in star_list if star.group == 0],
        color="red",
        marker="x",
    )
    ax.plot(
        [star.mass for star in star_list],
        [star.predict_fast(line_list) for star in star_list],
        color="red",
    )
    ax.scatter(
        [star.mass for star in star_list if star.group == 1],
        [star.period for star in star_list if star.group == 1],
        color="blue",
        marker="x",
    )
    ax.plot(
        [star.mass for star in star_list],
        [star.predict_slow(line_list) for star in star_list],
        color="blue",
    )
    figure


# def calculate_ser2(line_list, star_list):
#     return [
#         (line_list[1].predicted_value(star.mass) - star.period) ** 2
#         / (line_list[0].predicted_value(star.mass) - star.period) ** 2
#         for star in star_list
#     ]


# def calculate_weight_slow(line_list, star_list):
#     return np.divide(1, np.add(1, calculate_ser2(line_list, star_list)))


# def calculate_weight_fast(line_list, star_list):
#     return np.divide(1, np.add(1, np.divide(1, calculate_ser2(line_list, star_list))))


# def calculate_fitness(line_list, star_list):
#     tot = 0

#     w_f = calculate_weight_fast(line_list, star_list)
#     w_s = calculate_weight_slow(line_list, star_list)
#     for i, star in enumerate(star_list):
#         if star.group == 0:
#             tot += (line_list[0].predicted_value(star.mass) - star.period) ** 2 * w_f[i]
#         else:
#             tot += (line_list[1].predicted_value(star.mass) - star.period) ** 2 * w_s[i]

#     return tot


def switch_group(val):
    if val == 0:
        return 1
    else:
        return 0


def update_slow_line(star_list, line_list):
    """
    Calculates slope and intercept of the SLOW line, taking into account
    the weightings of the points compared to the other line, the further away
    the point from the line, the less it affects the fit.
    Inputs
        star_list: List of Star objects, seperate this input list into group
            one or group zero

    Retuns
        slope: slope of the slow line
        intercept: intercept of the slow line
    """
    linreg = LinearRegression()
    linreg.fit(
        [[star.mass] for star in star_list if star.group == 1],
        [star.period for star in star_list if star.group == 1],
        # sample_weight=[
        #     star.calculate_weight_s(line_list) for star in star_list if star.group == 1
        # ],
    )
    slope = linreg.coef_[0]
    intercept = linreg.intercept_

    return [slope, intercept]


def update_fast_line(star_list, line_list):
    """
    Calculates slope and intercept of the FAST line, taking into account
    the weightings of the points compared to the other line, the further away
    the point from the line, the less it affects the fit.
    Inputs
        star_list: List of Star objects, seperate this input list into group
            one or group zero

    Retuns
        slope: slope of the slow line
        intercept: intercept of the slow line
    """
    linreg = LinearRegression()
    linreg.fit(
        [[star.mass] for star in star_list if star.group == 0],
        [star.period for star in star_list if star.group == 0],
        # sample_weight=[
        #     star.calculate_weight_f(line_list) for star in star_list if star.group == 0
        # ],
    )
    slope = 0  # linreg.coef_[0]
    intercept = linreg.intercept_

    return [slope, intercept]


#%% DATA INITIALISATION AND PLOTING
path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

# importation of star objects
star_list = get_data(path)
# -6.99247391456971 11.866198098140732
# 0.17567270713813296 1.2806533526705661
# line_list = [
#     Line(0.000000, 1.2806533526705661),
#     Line(-6.99247391456971, 11.866198098140732),
# ]
line_list = [[0.000000, 2.2806533526705661], [-6.99247391456971, 11.866198098140732]]

for star in star_list:
    star.update_group(line_list)

plot_data(star_list, line_list)

#%%
figure1, ax1 = plt.subplots(1, figsize=(10, 8))
ax1.invert_xaxis()
ax1.scatter(
    [star.mass for star in star_list],
    [star.period for star in star_list],
    c=[star.calculate_weight_f(line_list) for star in star_list],
    cmap="coolwarm",
)

ax1.plot(
    [star.mass for star in star_list],
    [star.predict_slow(line_list) for star in star_list],
    color="blue",
)
ax1.plot(
    [star.mass for star in star_list],
    [star.predict_fast(line_list) for star in star_list],
    color="red",
)

#%%
print(line_list[0])
plot_data(star_list, line_list)

line_list[0] = update_fast_line(star_list, line_list)
line_list[1] = update_slow_line(star_list, line_list)

print(line_list[0])
plot_data(star_list, line_list)


for star in star_list:
    star.update_group(line_list)


#%%
