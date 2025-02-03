# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:18:59 2024

@author: sarat
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import os
pi=np.pi

#%% fit functions ###            

# def dampedGaussian(x,C,T,f,a,b):                         
#     return C * np.exp(-(x/T)**2/2)*(np.sin(2*pi*f*x+a))+b 

def dampedContrast(x,C,T,b):
    return C *np.exp(-(x/T)**2/2)+b  


   
def devSt(tau,d0,sigma,phi):
    a = 0.5 * ( 1 + np.cos(2*d0*tau+2*phi)*np.exp(-2*sigma**2*tau**2)) - (np.cos(d0*tau+phi))**2*np.exp(-sigma**2*tau**2)
    return np.sqrt(a)

def devSt_time(tau,omega0,omega1,sigma):
    d0 = omega0 + omega1*tau
    a = 0.5 * ( 1 + np.cos(2*d0*tau)*np.exp(-2*sigma**2*tau**2)) - (np.cos(d0*tau))**2*np.exp(-sigma**2*tau**2)
    return np.sqrt(a)


    

#%% Import

location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Export'

file = os.path.join(location, 'RamseyAv_Points.txt')
time, popMean, popStd = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#', unpack=True)

file = os.path.join(location, 'Ramsey_fitParams.txt')
poptRamsey = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
                                 
file = os.path.join(location, 'MaxPoints_fitParams.txt')
poptMaxPoints = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#', unpack=True)

file = os.path.join(location, 'C_fit.txt')
C_fit = np.genfromtxt(file, delimiter='\t',  skip_header=1, comments='#', unpack=True)
                                  
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\Monte Carlo\Export'
file = os.path.join(location, 'MCErrors_stdS.txt')
MC_stdErrors = np.genfromtxt(file, delimiter='\t', skip_header=1, comments='#',unpack=True)



#%% fitting of devSt[S] to infer Delta(delta0)

T = 26e-6                                                      # pi/2 pulse 
timePlot = np.linspace(10*10**(-6),3.46*10**(-3), num=1200)    # tau : waiting time
timePlot += T*4/np.pi


max_points_fit = dampedContrast(time,*poptMaxPoints)
contrast_max = max_points_fit -0.5                    # contrast decay given by B field inhomogeneities 
d0 = poptRamsey[1]                                       # rad/s : detuning from Ramsey fit 
sigma_d0 = 1/poptRamsey[0]                                    # rad/s

model = lm.Model(devSt_time)
params = model.make_params()
params['omega0'].set(value=poptRamsey[1], vary=False)
params['omega1'].set(value=poptRamsey[2], vary=False)
# params['omega_p'].set(value=poptRamsey[3], vary=False)
# params['phi'].set(value=poptRamsey[4], vary=False)
params['sigma'].set(value=sigma_d0, vary=True)


#fitting function is sts_S/C
y = popStd/contrast_max
err_y = MC_stdErrors/contrast_max                        # errors on the std from Monte Carlo simulations
weights = 1 / err_y            
out = model.fit(y, params=params, tau=time)
#out = model.fit(y, params=params, tau=time,weights=weights)
stdS_fit = model.eval(out.params, tau = timePlot)

print()
print("stdS fit:")
print(out.fit_report())

print()
print("B field noise from sigma:")
print(f"{(out.params['sigma'].value)/(2*pi*700000):.3e} +/- {((out.params['sigma'].stderr))/(2*pi*700000):.3e} G")
print(f"{(out.params['sigma'].value)/(2*pi):.3e} +/- {((out.params['sigma'].stderr))/(2*pi):.3e} Hz")

#%% plots

size = 14
colors = plt.get_cmap('Paired').colors

######### plot std[S] exact fit
figStdS, axStdS =  plt.subplots(1,1,constrained_layout=True)

axStdS.errorbar(time*1e3, popStd, yerr=MC_stdErrors,
                fmt='o', markersize=4.4, markeredgewidth=1, elinewidth=1.1,
                mec=colors[4], mfc=colors[4], ecolor=colors[4])


axStdS.plot(timePlot*1e3, stdS_fit*(C_fit-0.5),
           '-', color = 'dimgrey',linewidth = '1.2')

#res = devSt_time(timePlot, poptRamsey[1], poptRamsey[2], poptRamsey[3], 1/0.00275892, poptRamsey[4])
# axStdS.plot(timePlot*1e3, res*(C_fit-0.5),
#            '-', color = 'orange',linewidth = '1.2')

# refinements
axStdS.set_ylabel(r"$\sigma_{P_{|\uparrow\rangle}}$", size=size)
axStdS.set_xlabel(r"waiting time $\tau$ (ms)", size=size)
axStdS.tick_params(axis='both', which='major', labelsize=size)
# add box
fit_params_text = "\n".join([f"$\sigma$={out.params['sigma'].value/2/pi:.0f} Hz"])
axStdS.text(0.05, 0.95, fit_params_text, transform=axStdS.transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1, facecolor='lightgrey'))



#%% Export
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Export'


outarray = np.vstack((time, popStd, MC_stdErrors)).T
header = 'time \t stdS \t MC_stdErrors'
np.savetxt(os.path.join(location, 'StdS_data.txt'), outarray, header=header, delimiter='\t')

outStdSFit_Params = [out.params['sigma'].value]
header = 'sigma (Hz)'
np.savetxt(os.path.join(location, 'StdS_fitParams.txt'), outStdSFit_Params, header=header, delimiter='\t')


outStdS_fit = stdS_fit*(C_fit-0.5)
header = 'stdS_fit'
np.savetxt(os.path.join(location, 'StdS_fit.txt'), outStdS_fit, header=header, delimiter='\t')


# outStdS_fit_T2 = res*(C_fit-0.5)
# header = 'stdS_fit_T2'
# np.savetxt(os.path.join(location, 'StdS_fit_T2.txt'), outStdS_fit_T2, header=header, delimiter='\t')

#%% Save figures
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Figures'

# figStdS.savefig(os.path.join(location, 'StdS.pdf'), dpi=600)

