import numpy as np
import csv
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg

with open("Bowen_test.csv",'r') as i:           #open a file in directory of this script for reading 
    rawdata = list(csv.reader(i,delimiter=","))   #make a list of data in file
    
exampledata = np.array(rawdata[1:], dtype= np.float64)    #convert to data array
xdata = exampledata[:,0]
ydata = exampledata[:,1]

plt.figure(1,dpi=120)
plt.yscale('linear')
plt.xscale('linear')
#plt.xlim(0,4)
#plt.ylim(0,2.5)
plt.title("Example Data")
plt.xlabel(rawdata[0][0])
plt.ylabel(rawdata[0][1])
plt.plot(xdata,ydata,label="Experimental Data")

def func(x,b):                                   #input x in nm and b in nm^-1
    return a0 * np.exp(-b * x) + a1

a0 = 2.5    #W m^-2 nm^-1
a1 = 0.5    #W m^-2 nm^-1

funcdata = func(xdata,1.375)                 #Generate & Plot data for comparison
plt.plot(xdata,funcdata,label="Model")   
plt.legend()

popt, pcov = curve_fit(func,xdata,ydata,bounds=(0,4))
perr = np.sqrt(np.diag(pcov))

TotalInt = intg.trapz(ydata,xdata)                     #Compute numerical integral
TotalInt_func = intg.quad(func,0,4, args=(1.375))[0]        #Compute integral of function
low_Frac = intg.quad(func,0,2, args=(1.375))[0]/TotalInt_func
high_Frac = intg.quad(func,2,4, args=(1.375))[0]/TotalInt_func

plt.show()