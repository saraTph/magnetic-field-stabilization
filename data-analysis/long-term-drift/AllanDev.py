# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:12:55 2024

@author: sarat
"""

import numpy as np
import datetime
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import allantools


#%% fit functions ###            


#%% import data
# 10_33_40 → 11_59_29
f_y = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\long term drift\AllanDev_data'

# create time vector
# Start and end times
start_time = datetime.datetime.strptime("09:58:40", "%H:%M:%S")
end_time = datetime.datetime.strptime("13:38:29", "%H:%M:%S")

# Generate 600 equally spaced datetime points between start and end
time = [start_time + datetime.timedelta(seconds=s) 
               for s in np.linspace(0, (end_time - start_time).total_seconds(), 1200)]

pop50 = scipy.io.loadmat(f_y)['pop1V']


#%% Allan Deviation

allanDev = allantools.oadev(pop50, taus = 'all')

#%% Plot

label_plot = 'Allan Dev'
title_plot = 'allan dieviation'


#### plot averaged data
figM, axM = plt.subplots(1,1,constrained_layout=True, figsize = (10,5))
axM.errorbar(allanDev[0], allanDev[1], yerr = allanDev[2], color = 'blue' ,markersize = '2')
# refinements
size = 14
axM.set_ylabel(label_plot, size=size)
axM.tick_params(axis='both', which='major', labelsize=size)
axM.set_title(title_plot, size=size)
axM.tick_params(axis='x', rotation=50, labelsize=10) 

