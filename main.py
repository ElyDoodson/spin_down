#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress

#%%


class Star:
    def __init__(self, period, mass):
        self.period = period
        self.mass = period

    def set_initial_group(self):
        self.group = 1 if calculate_line(0, self.mass, 7) > self.period else 0


def calculate_line(m, x, c):
    return np.add(np.dot(m, x), c)


#%%
