#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = mass

    def set_initial_group(self):
        self.group = 1 if self.period >= calculate_line(0, self.mass, 7) else 0


class Line:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def updateforstars_true(self, star_list):
        """
        this updates the slope of a line for the true slope and intercept
        """
        slope_s, intercept_s = linregress(
            [star.mass for star in star_list if star.group == 1],
            [star.period for star in star_list if star.group == 1],
        )[:2]
        self.slope = slope_s
        self.intercept = intercept_s

    def updateforstars_average(self, star_list):
        """
        This updates the slope of the line using the average of the data.
        Effectively fixing the slope at 0 and only changing its intercept
        """
        slope_f, intercept_f = (
            0,
            np.sum([star.period for star in star_list if star.group == 0])
            / len([star.group for star in star_list if star.group == 0]),
        )
        self.slope = slope_f
        self.intercept = intercept_f

    def predicted_value(self, star):
        return calculate_line(self.slope, star, self.intercept)


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


def get_lines(star_list):
    """
    Takes in list of star objects and finds the initial lines
    """
    slope_s, intercept_s = linregress(
        [star.mass for star in star_list if star.group == 1],
        [star.period for star in star_list if star.group == 1],
    )[:2]
    slope_f, intercept_f = (
        0,
        np.sum([star.period for star in star_list if star.group == 0])
        / len([star.group for star in star_list if star.group == 0]),
    )

    return [Line(slope_f, intercept_f), Line(slope_s, intercept_s)]


def plot_data(star_list, line_list, figure="figure", axes="ax"):
    figure, ax = plt.subplots(1, figsize=(10, 8))
    ax.invert_xaxis()

    ax.scatter(
        [star.mass for star in star_list if star.group == 0],
        [star.period for star in star_list if star.group == 0],
        color="green",
    )
    ax.plot(
        [star.mass for star in star_list],
        [line_list[0].predicted_value(star.mass) for star in star_list],
    )
    ax.scatter(
        [star.mass for star in star_list if star.group == 1],
        [star.period for star in star_list if star.group == 1],
        color="red",
    )
    ax.plot(
        [star.mass for star in star_list],
        [line_list[1].predicted_value(star.mass) for star in star_list],
    )
    pass


path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"

# importation of star objects
star_list = get_data(path)
# sets all the initial star groups to an arbitrary line
for star in star_list:
    star.set_initial_group()

# initialisation of lines
line_list = get_lines(star_list)


plot_data(star_list, line_list, figure1, ax1)

#%%