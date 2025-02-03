# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:35:08 2024

@author: Sarah

plot the results of Ramsey analysis for the paper
"""

import numpy as np
import os
import matplotlib.pyplot as plt

#%% Import

location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Export'


#import Ramsey raw data
file = os.path.join(location, 'Ramsey_allPoints.txt')
pop0, pop1, pop2, pop3, pop4, pop5, pop6, pop7, pop8, pop9 = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
popTab = np.vstack((pop0, pop1, pop2, pop3, pop4, pop5, pop6, pop7, pop8, pop9))
#import Ramsey 10th curve
file = os.path.join(location, 'Ramsey10_cut.txt')
scan10_cut, pop10_cut = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#', unpack=True)   
            
               
            
# import analyzed data
file = os.path.join(location, 'C.txt')                                                                       #import C max points
C = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)                                

file = os.path.join(location, 'RamseyAv_Points.txt')                                                         #import Ramsey mean data and std
time, popMean, popStd = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#', unpack=True) 
                                  
file = os.path.join(location, 'StdS_data.txt')                                                               #import StdS data and errorbar
time, stdS, MC_stdErrors = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
     
                            

# import fits
file = os.path.join(location, 'C_fit.txt')
C_fit = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
                                  
file = os.path.join(location, 'C_Gaussianfit.txt')
C_GaussianFit = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#', unpack=True)

file = os.path.join(location, 'Ramsey_fit.txt')
Rmasey_fit = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
                                  
file = os.path.join(location, 'StdS_fit.txt')                                                                  # stdS_fit
stdS_fit = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)

file = os.path.join(location, 'StdS_fit_T2.txt')                                                                  # stdS_fit
stdS_fit_T2 = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
                                 


#%% def variables 
T = 26e-6  
timePlot = np.linspace(10*10**(-6),3.46*10**(-3), num=1200) #waiting time
timePlot = timePlot + T*4/np.pi

# %%

fig, ax = plt.subplots(2,1,constrained_layout=True, figsize = (6.5,7.5))
plt.rcParams['font.family'] = 'Times New Roman'
############################################################# all points fit


#col = '#DB7C26'
col = '#80A4ED'
size = 16

for i in range(10):
    if i !=9:
        ax[0].errorbar(time*1e3, popTab[i,:], yerr=None, 
                     fmt='o', markersize=4, markeredgewidth=0.7, elinewidth=0.7,
                     mec=col, mfc='w', ecolor=col,alpha=1, zorder = 1)
    else:
        ax[0].errorbar(scan10_cut, pop10_cut, yerr=None, 
                     fmt='o', markersize=4, markeredgewidth=1, elinewidth=1,
                     mec=col, mfc='w', ecolor=col,alpha=1, zorder = 1)
ax[0].plot(timePlot*1e3, C_fit, '-', 
           color  =col, linewidth = '1.8')

ax[0].plot(timePlot*1e3, C_GaussianFit, ':', 
           color = col, linewidth = '1.8')


############################################################ mean curve plot data + fit

col = '#283044'


ax[0].errorbar(time*1e3, popMean, yerr=popStd, fmt='o', 
               markersize=4, markeredgewidth=1, elinewidth=1,
               mec=col, mfc=col, ecolor=col, zorder = 3)


ax[0].plot(timePlot*1e3, Rmasey_fit, '-', 
           color = col, linewidth = '1.8', zorder = 3)


#reifiniments 
ax[0].set_ylim([0, 1])
ax[0].xaxis.set_tick_params(labelbottom=False) # Turn off x-axis labels for the top plot
ax[0].set_ylabel(r"$f$", size=18)
ax[0].tick_params(axis='both', which='major', labelsize=size)



############################################################ stdS fit

ax[1].errorbar(time*1e3, popStd, yerr=MC_stdErrors,
                fmt='o', markersize=4, markeredgewidth=1, elinewidth=1,
                mec=col, mfc=col, ecolor=col)

ax[1].plot(timePlot*1e3, stdS_fit,
           '-', color = col, linewidth = '1.8')


# ax[1].plot(timePlot*1e3, stdS_fit_T2,
#            '-', color = 'red',linewidth = '1.8', label = r'$\sigma = 1/T_2 = 55(3)$ Hz')

# refinements
ax[1].set_ylim([-0.05, 0.37])
ax[1].set_ylabel(r"Std[$f$ ]", size=18)
ax[1].set_xlabel(r"Ramsey time $T$ (ms)", size=18)
ax[1].tick_params(axis='both', which='major', labelsize=size)
#ax[1].legend(loc='upper left',fontsize=12)

# Add label (a) outside the top-left of the first plot
fig.text(0.02, 0.95, '(a)', ha='center', va='center', fontsize=18, 
         transform=fig.transFigure)

# Add label (b) outside the top-left of the second plot
fig.text(0.02, 0.49, '(b)', ha='center', va='center', fontsize=18, 
         transform=fig.transFigure)

# # Set y-tick labels every 0.5 units
# y_ticks = np.arange(ax[1].get_ylim()[0], ax[1].get_ylim()[1], 0.05)
# ax[1].set_yticks(y_ticks)
# # add box
# fit_params_text = "\n".join([f"$\sigma$={poptStdS[1]/(2*np.pi):.0f} Hz"])
# ax[1].text(0.05, 0.95, fit_params_text, transform=ax[1].transAxes, 
#          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1, facecolor='lightgrey'))

#%% Save figures
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Figures'

fig.savefig(os.path.join(location, 'Ramsey_DV.pdf'), dpi=600)

