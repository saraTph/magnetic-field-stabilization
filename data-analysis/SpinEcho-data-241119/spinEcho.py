# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:10:41 2024

@author: Sarah
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import scipy.io
import os

pi=np.pi
#fsolve=scipy.optimize.fsolve

#%% fit functions ###            


def cos_var(x,a,C,T,omega0,omega1):
    # f = omega0 + omega1*np.cos(omega_p * x + phi)
    f = omega0 + omega1*x**2
              
    return a + C*np.exp(-(x/T)**2/2)*(np.sin(f*x+np.pi/2)) 
 

#%% import data from matlab and store them


f_y = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241119-SpinEcho\spin_echo.mat'


echo = scipy.io.loadmat(f_y)[f'pop1V'].T[0]

time = np.linspace(10*10**(-6),10.93*10**(-3), num=92)          #waiting time
T = 2*time 

timePlot = np.linspace(10*10**(-6),10.93*10**(-3), num=1200)          #waiting time
TPlot = 2*timePlot

#%%
model = lm.Model(cos_var)
params_C = model.make_params()
params_C['a'].set(value=0.5, vary=False)
params_C['C'].set(value=0.41, vary=False)
params_C['T'].set(value=8.9e-3, vary=False)
params_C['omega0'].set(value=120*2*np.pi, vary=True)
params_C['omega1'].set(value=600*2*np.pi, vary=True)
# params_C['omega_p'].set(value=(300)*2*np.pi, vary=False)
# params_C['phi'].set(value=1, vary=True)

y = echo
out = model.fit(y, params=params_C, x=T)
#out_C = model.fit(y, params=params_C, x=time)

#datafit = model.eval(out.params, x=TPlot)
fitPlot = model.eval(out.params, x=TPlot)

print(out.fit_report())

#%%

colors = plt.get_cmap('Paired').colors
size = 14

figRam, axRam = plt.subplots(1,1,constrained_layout=True)
axRam.plot(T*1e3, y, 'o', 
           color = 'dimgrey',linewidth = '1.3')

axRam.plot(TPlot*1e3, fitPlot, '-', 
           color = 'dimgrey',linewidth = '1.3')



# refinements
size = 14
axRam.set_ylim([0, 1])
axRam.set_ylabel(r"P$_{|\uparrow\rangle}$", size=size)
axRam.set_xlabel(r"waiting time $\tau$ (ms)", size=size)
axRam.tick_params(axis='both', which='major', labelsize=size)


# x = np.linspace(10*10**(-6),3.46*10**(-3), num=1200)
# # phi = poptRamsey[4]
# # f = poptRamsey[1] + poptRamsey[2]*np.cos(poptRamsey[3] * x + phi) 
# f = out.params['omega0'].value + out.params['omega1'].value * x
# S = -np.cos(f*x)

# print(out.params['omega0'].value/2/np.pi)
# print(out.params['omega1'].value/2/np.pi)

# fig, ax = plt.subplots(1,1,constrained_layout=True)
# #ax.plot(x*1e3,f/(2*np.pi),'o')
# ax.plot(x*1e3,f/(2*np.pi),'-',linewidth = '1.8')
# ax.set_ylabel(r"$\omega/(2\pi)$", size=14)
# ax.set_xlabel(r"waiting time $\tau$ (ms)", size=14)
# ax.tick_params(axis='both', which='major', labelsize=14)