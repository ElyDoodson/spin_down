#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress



class Star:
    slope_slow = 0
    intercept_slow= 0
    slope_fast = 0
    intercept_fast = 0
    
    def __init__(self, period, mass):
        self.period = period
        self.mass = mass
        #abritrary horizontal line, set to (-5,,7) for rough equal data split 
        self.group = 1 if self.period >= calculate_line(0, self.mass, 7) else 0
        self.predict_fast = calculate_line(Star.slope_fast, self.mass, Star.intercept_fast)
        self.predict_slow = calculate_line(Star.slope_slow, self.mass, Star.intercept_slow)
        self.ser = square_err(self.period, self.predict_fast, self.predict_slow)
        self.weight_slow = 1 / (1 + self.ser )
        self.weight_fast = 1 / (1 + (1 / self.ser))
        
        
def get_data(location):
    """
    Takes string as input and returns a list of Star objects
    """
    data_frame = pd.read_csv(location, encoding = "utf-8", delimiter = "\t", comment = "#")

    mass = data_frame.M.tolist()
    period = data_frame.Per.tolist()
    
    return [Star(period, mass) for mass, period in zip(mass,period)]


def calculate_line(m,x,c):
    return np.add(np.dot(m,x), c)

def calculate_fits(star_list):
    
    slope_s, intercept_s = linregress([star.mass for star in star_list if star.group == 1],[star.period for star in star_list if star.group == 1])[:2]
    slope_f, intercept_f = 0, np.sum([star.period for star in star_list if star.group == 0])/len([star.group for star in star_list if star.group == 0])
#     slow_slope, slow_intercept = linregress([star.mass for star in star_list if star.group == 0],[star.period for star in star_list if star.group == 0])[:2]
    

    Star.slope_fast = slope_f
    Star.intercept_fast = intercept_f
    
    Star.slope_slow = slope_s
    Star.intercept_slow = intercept_s
    
def square_err(period, predict_fast, predict_slow):
    return (predict_slow - period)**2 / (predict_fast - period)**2

def calculate_mse(star_list):
    tot = 0
    for star in star_list:
        if star.group == 1:
            tot += (star.predict_slow - star.period)**2 * star.weight_slow
        else:
            tot += (star.predict_fast - star.period)**2 * star.weight_fast
#         tot += ((star.predict_fast - star.period)**2 *star.weight_fast) if star.group == 0 else ((star.predict_slow - star.period)**2 *star.weight_slow)
    return tot

def switch_group(datum):
    if datum == 1:
        return 0
    else:
        return 1


# In[24]:


path = "d:data\Pleiades_Hartman.csv"
# path = "/home/edoodson/Documents/spin_down/data/Pleiades_Hartman.csv"


star_list = get_data(path)

calculate_fits(star_list)

figure1, ax1 = plt.subplots(1, figsize = (10,8))
ax1.invert_xaxis()
ax1.set(ylim = (0,20))
ax1.scatter([star.mass for star in star_list], [star.period for star in star_list], c=[star.weight_fast for star in star_list], cmap = "coolwarm")
ax1.plot([star.mass for star in star_list], [star.predict_fast for star in star_list], color = "red")
ax1.plot([star.mass for star in star_list], [star.predict_slow for star in star_list])


# In[25]:



for i in range(1):
    for star in star_list:
        calculate_fits(star_list)
        print("bfr:",star.slope_slow)
        initial_mse = calculate_mse(star_list)
        
        star.group = switch_group(star.group)
        
        calculate_fits(star_list)
        print("aft:",star.slope_slow)
        print()
        final_mse = calculate_mse(star_list)
        
#         print("initial:", initial_mse, " ", "final:",final_mse)
        if final_mse > initial_mse:
            star.group = switch_group(star.group)

figure1, ax1 = plt.subplots(1, figsize = (10,8))
ax1.invert_xaxis()
ax1.set(ylim = (0,20))
ax1.scatter([star.mass for star in star_list], [star.period for star in star_list], c=[star.weight_slow for star in star_list], cmap = "coolwarm")
ax1.plot([star.mass for star in star_list], [star.predict_fast for star in star_list] )
ax1.plot([star.mass for star in star_list], [star.predict_slow for star in star_list])


# In[ ]:




