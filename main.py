import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = period

    def set_initial_group(self):
        self.group = 1 if calculate_line(0, self.mass, 7) > self.period else 0


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


path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"


star_list = get_data(path)
for star in star_list:
    star.set_initial_group()
