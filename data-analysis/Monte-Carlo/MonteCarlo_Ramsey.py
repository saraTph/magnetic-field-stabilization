# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:42:30 2024

@author: Sarah
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import lmfit as lm
pi = np.pi


#%% fit functions ###            

# def dampedGaussian(x,C,T,f,a,b):                         
#     return C * np.exp(-(x/T)**2/2)*(np.sin(2*pi*f*x+a))+b 

def dampedContrast(x,C,T,b):
    return C *np.exp(-(x/T)**2/2)+b  

def devStS_solved(tau,d0,sigma): 
    return np.sqrt(0.5+0.5*np.exp(-2*sigma**2*tau**2)*np.cos(2*d0*tau)-(np.cos(d0*tau))**2*np.exp(-tau**2*sigma**2))


#%% Import

location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241024\Export'

file = os.path.join(location, 'RamseyAv_Points.txt')
time, popMean, popStd = np.genfromtxt(file, delimiter='\t', 
                                  skip_header=1, comments='#',
                                  unpack=True)
file = os.path.join(location, 'Ramsey_fitParams.txt')
poptRamsey = np.genfromtxt(file, delimiter='\t', 
                                  skip_header=1, comments='#',
                                  unpack=True)
file = os.path.join(location, 'MaxPoints_fitParams.txt')
poptMaxPoints = np.genfromtxt(file, delimiter='\t', 
                                  skip_header=1, comments='#',
                                  unpack=True)

#%% Monte Carlo simulation

# Parameters
C = dampedContrast(time,*poptMaxPoints)-0.5          # contrast decay given by B field inhomogeneities
d0 = poptRamsey[2]*(2*pi)       # middle value detuning d                     
sigma = 150 * 2*pi                     # Standard deviation of Gaussian noise
num_samples = 2             # Number of samples for Monte Carlo simulation


# Initialize array to store results
S_mean = np.zeros(len(time))
S_std = np.zeros(len(time))

# Monte Carlo simulation loop for each time in tau
for j, t in enumerate(time):
    # Array to store S values at current time t
    S_values = np.zeros(num_samples)
    
    # Monte Carlo simulation for the current time point
    for i in range(num_samples):
        # Generate Gaussian noise for Delta(d0)
        delta_d0 = np.random.normal(0, sigma)
        
        # Calculate S using the noisy value at current time t
        S = 0.5 + C[j] * np.sin((d0 - delta_d0) * t-pi/2)
        S_values[i] = S 
        
    
    # Calculate mean and standard deviation of S at time t
    S_mean[j] = np.mean(S_values)
    S_std[j] = np.std(S_values)

#%% StdS fit
model = lm.Model(devStS_solved)
params = model.make_params()
params['d0'].set(value=d0, vary=False)
params['sigma'].set(value=sigma, vary=True)
y = S_std/C
out = model.fit(y, params=params, tau=time)


#%% plots

size = 14

######### plot std in the linear regime
figMC, axMC = plt.subplots(1,1,constrained_layout=True)
axMC.plot(time*1e3, S_mean, label='Mean of S', color='blue')
axMC.fill_between(time*1e3, S_mean - S_std, S_mean + S_std, color='blue', alpha=0.3, label='Standard deviation of S')
# refinements
axMC.set_title('Monte Carlo Simulation of S', size=size)
axMC.set_ylabel('S', size=size)
axMC.set_xlabel(r"$\tau$ [ms]", size=size)
axMC.tick_params(axis='both', which='major', labelsize=size)
axMC.legend()

######### plot std[S] exact fit
figStdS, axStdS =  plt.subplots(1,1,constrained_layout=True)
axStdS.scatter(time*1e3, y)
#axStdS.plot(time*1e3, outP.init_fit, 'darkorange', label=r'$\tau+\tau_{\pi/2}$')
axStdS.plot(time*1e3, out.best_fit, 'darkorange')
# refinements
axStdS.set_title(f'Monte Carlo simulation {num_samples} samples')
axStdS.set_ylabel(r"std[S]", size=size)
axStdS.set_xlabel(r"$\tau$ [ms]", size=size)
axStdS.tick_params(axis='both', which='major', labelsize=size)
axStdS.legend()
# add box
param_names = [r'$\delta_0$', r'$\sigma_{\delta_0}$'] 
fit_params_text = "\n".join([
    f"{new_name}: {value.value/(2*pi):.0f} Hz" 
    for new_name, (name, value) in zip(param_names, out.params.items())
])
axStdS.text(0.05, 0.95, fit_params_text, transform=axStdS.transAxes, 
        fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1, facecolor='lightgrey'))


#%% Save figures
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\Monte Carlo\Figures'

#figMC.savefig(os.path.join(location, 'MC_ramsey.png'), dpi=600)
#figStdS.savefig(os.path.join(location, 'MC_srdS.png'), dpi=600)
