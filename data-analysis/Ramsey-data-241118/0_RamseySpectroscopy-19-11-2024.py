# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:07:55 2024

@author: Sarah

descrizzzz
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import scipy.io
import os

pi=np.pi
#fsolve=scipy.optimize.fsolve

#%% fit functions ###            

def dampedGaussian(x,C,T,f,a,b):                         
    return b - C*np.exp(-(x/T)**2/2)*(np.cos(f*x+a)) 

def dampedGaussian_C(x,T,f,a):                         
    return -np.exp(-(x/T)**2/2)*(np.cos(f*x+a)) 

def dampedGaussian_var(x,T,omega0,omega1):
    # f = omega0 + omega1*np.cos(omega_p * x + phi)
    f = omega0 + omega1*x
              
    return -np.exp(-(x/T)**2/2)*(np.cos(f*x)) 

def dampedContrast(x,C,T,b):
    return b + C *np.exp(-(x/T)**2/2)  

#%% import data from matlab and store them

f_x = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\RamseyScans.mat'
f_y = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\RamseyPop.mat'

Data= {}
for i in range(9):
    if not i in Data.keys():
        Data[i] = {}
    sorter = np.argsort(scipy.io.loadmat(f_x)[f'scan{i+1}'][0])
    Data[i]['scan'] = scipy.io.loadmat(f_x)[f'scan{i+1}'][0][sorter]
    Data[i]['pops'] = scipy.io.loadmat(f_y)[f'pop{i+1}'].T[0][sorter]
    
#add the 10th Ramsey curve with only 68 points
sorter = np.argsort(scipy.io.loadmat(f_x)[f'scan{10}'][0])
scan10_cut = scipy.io.loadmat(f_x)[f'scan{10}'][0][sorter]
pop10_cut = scipy.io.loadmat(f_y)[f'pop{10}'].T[0][sorter]

#I have two missing points in scan10 at positions 14 and 53
scan10 = scan10_cut
scan10 = np.insert(scan10, 14, 0.71)
scan10 = np.insert(scan10, 53, 2.66)
#add two fake data = 0 at positions 14 and 53
pop10 = pop10_cut
pop10 = np.insert(pop10, 14, 0)
pop10 = np.insert(pop10, 53, 0)
#add them to Data struct
Data[10] = {}
Data[10]['scan'] = scan10
Data[10]['pops'] = pop10


popTab = np.array([Data[key]['pops'] for key in Data.keys()]) 
popMean = np.zeros(70)
popStd = np.zeros(70)
for i in range(70):
    if i == 14 or i==53:
        popMean[i] = np.mean(popTab[:9,i])
        popStd[i] = np.std(popTab[:9,i])
    else:
        popMean[i] = np.mean(popTab[:,i])
        popStd[i] = np.std(popTab[:,i])
        
T = 26e-6                                                      #pulse duration
time = np.linspace(10*10**(-6),3.46*10**(-3), num=70)          #waiting time
timePlot = np.linspace(10*10**(-6),3.46*10**(-3), num=1200)    #waiting time for fit plots
time += T*4/pi
timePlot += T*4/pi


#%% 

########## fit of the max points C_max ######################################
    
all_peaks_x = []
all_peaks_y = []
max_index = (14,24,43,53,63)

for i in range(len(max_index)):
    all_peaks_y.append(np.max(popTab[:,max_index[i]]))
    all_peaks_x.append(time[max_index[i]])    
    
      
model = lm.Model(dampedContrast)
paramsC = model.make_params()
paramsC['C'].set(value=0.42, vary=True)
paramsC['T'].set(value=2*1e-3, vary=True)
paramsC['b'].set(value=0.5, vary=True)   
outC = model.fit(all_peaks_y, params=paramsC, x=all_peaks_x)

C_Max =  model.eval(outC.params, x=time)
C_Plot = model.eval(outC.params, x=timePlot)

contrastMax = C_Max-0.5

print()
print("Max points fit:")
print(outC.fit_report())



########### fit the averaged curve  #########################################

#dampedGaussian_var(x,T,f0,f1,omega_p, phi,a):

    

model = lm.Model(dampedGaussian_var)
params_C = model.make_params()
params_C['T'].set(value=2.8*1e-3, vary=True)
params_C['omega0'].set(value=12738.8041, vary=True)
params_C['omega1'].set(value=304.384508, vary=True)
# params_C['omega_p'].set(value=(300)*2*np.pi, vary=False)
# params_C['phi'].set(value=1, vary=True)

y = (popMean-0.5)/contrastMax
out_C = model.fit(y, params=params_C, x=time, weights=1/popStd)
#out_C = model.fit(y, params=params_C, x=time)

RamseyMean_fit = model.eval(out_C.params, x=timePlot)
RamseyMean_fitPlot = model.eval(out_C.params, x=timePlot)

print()
print("Averaged Ramsey fit:")
print(out_C.fit_report())

print()
print("B field noise from T2:")
print(f"{1/(out_C.params['T'].value*(2*np.pi)*700000):.3e} +/- {1/((out_C.params['T'].value-out_C.params['T'].stderr)*2*np.pi*700000)-1/(out_C.params['T'].value*(2*np.pi)*700000):.3e} G" )
print(f"{1/(out_C.params['T'].value*(2*np.pi)):.3e} +/- {1/((out_C.params['T'].value-out_C.params['T'].stderr)*2*np.pi)-1/(out_C.params['T'].value*(2*np.pi)):.3e} Hz" )




#%% Plot

colors = plt.get_cmap('Set2').colors
colors = plt.get_cmap('Paired').colors
size = 14


##### Plot max contrast
figC, axC = plt.subplots(1,1,constrained_layout=True)
for i, keyi in enumerate(Data.keys()):
    if i !=9:
        axC.errorbar(time*1e3, Data[keyi]['pops'],yerr=None, 
                     fmt='o', markersize=3.7, markeredgewidth=1.1, elinewidth=1.1,
                     mec=colors[3], mfc=colors[2], ecolor=colors[2])
    else:
        axC.errorbar(scan10_cut, pop10_cut,yerr=None, 
                     fmt='o', markersize=3.7, markeredgewidth=1, elinewidth=1.2,
                     mec=colors[3], mfc=colors[2], ecolor=colors[2])
axC.plot(timePlot*1e3, C_Plot, '-', 
         color = 'dimgrey',linewidth = '1.5')
# axC.plot(timePlot*1e3, 1-C_Plot, '-', 
#          color = 'dimgrey',linewidth = '1.5')
# refinements
axC.set_ylim([0, 1])
axC.set_ylabel(r"P$_{|\uparrow\rangle}$", size=size)
axC.set_xlabel(r"waiting time $\tau$ (ms)", size=size)
axC.tick_params(axis='both', which='major', labelsize=size)


###### Plot Ramsey
figRam, axRam = plt.subplots(1,1,constrained_layout=True)
axRam.errorbar(time*1e3, popMean, yerr=popStd, fmt='o', 
               markersize=4.4, markeredgewidth=1.1, elinewidth=1.1,
               mec=colors[3], mfc=colors[2], ecolor=colors[3])

axRam.plot(timePlot*1e3, (RamseyMean_fitPlot)*(C_Plot-0.50)+0.5, '-', 
           color = 'dimgrey',linewidth = '1.3')


# d = dampedGaussian_var(timePlot,2.5*1e-3,12506,-(12506.4567-12945.5512),150*2*np.pi,0)
# axRam.plot(timePlot*1e3, d, '-', 
#          color = 'dimgrey',linewidth = '1.5')

# refinements
size = 14
axRam.set_ylim([0, 1])
axRam.set_ylabel(r"P$_{|\uparrow\rangle}$", size=size)
axRam.set_xlabel(r"waiting time $\tau$ (ms)", size=size)
axRam.tick_params(axis='both', which='major', labelsize=size)



#%% Export
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Export'


outarrayPop = np.vstack((popTab[0,:], popTab[1,:], popTab[2,:], popTab[3,:], popTab[4,:], popTab[5,:], popTab[6,:], popTab[7,:], popTab[8,:], popTab[9,:])).T
header = 'pop0 \t pop1 \t pop2 \t pop3 \t pop4 \t pop5 \t pop6 \t pop7 \t pop8 \t pop9'
np.savetxt(os.path.join(location, 'Ramsey_allPoints.txt'), outarrayPop, header=header, delimiter='\t')

outarrayPop_cut = np.vstack((scan10_cut, pop10_cut)).T
header = 'scan10_cut \t pop_10_cut'
np.savetxt(os.path.join(location, 'Ramsey10_cut.txt'), outarrayPop_cut, header=header, delimiter='\t')

outarray = np.vstack((time, popMean, popStd)).T
header = 'time \t mean \t st dev'
np.savetxt(os.path.join(location, 'RamseyAv_Points.txt'), outarray, header=header, delimiter='\t')

outRamseyFit_Params = [out_C.params['T'].value,
                       out_C.params['omega0'].value,
                       out_C.params['omega1'].value,
                       # out_C.params['omega_p'].value,
                       # out_C.params['phi'].value
                       ]
header = 'T \t f \t phase'
np.savetxt(os.path.join(location, 'Ramsey_fitParams.txt'), outRamseyFit_Params, header=header, delimiter='\t')

outMaxPointsFit_Params = [outC.params['C'].value, 
                          outC.params['T'].value,
                          outC.params['b'].value]
header = 'C \t T \t offset'
np.savetxt(os.path.join(location, 'MaxPoints_fitParams.txt'), outMaxPointsFit_Params, header=header, delimiter='\t')

outC = C_Max
header = 'C'
np.savetxt(os.path.join(location, 'C.txt'), outC, header=header, delimiter='\t')

outC_fit = C_Plot
header = 'C_Plot'
np.savetxt(os.path.join(location, 'C_fit.txt'), outC_fit, header=header, delimiter='\t')

outRamsey_fit = RamseyMean_fitPlot*(C_Plot-0.5)+0.5
header = 'RamseyFit_Plot'
np.savetxt(os.path.join(location, 'Ramsey_fit.txt'), outRamsey_fit, header=header, delimiter='\t')


#%% Save figures
location = r'C:\Users\Sarah\OneDrive\Documenti\InstOptique\Data analysis\B filed stabilization\data-241118\Figures'

# figRam.savefig(os.path.join(location, 'Ramsey.pdf'), dpi=600)
# figC.savefig(os.path.join(location, 'MaxPoints_fit.pdf'), dpi=600)