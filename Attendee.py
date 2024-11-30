import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from numpy import polyfit, poly1d
import os
from scipy import log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score

s = pd.read_csv('prod_test.csv')
print(s.head())
plt.scatter(s.account_user, s.cdf)
plt.show()


# p = polyfit(x,y,n)


def func(x, a, b):
#    y = a * log(x) + b
    y = x/(a*x+b)
    return y


x0 = [1,2,3,5,7,11,17,22,30,43,68,136,222,608]  
y0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

result = curve_fit(func, x0, y0,method='trf')
a, b = result[0]  

x1 = np.arange(1, 225, 0.1)  
y1 = x1/(a*x1+b)

x0 = np.array(x0)
y0 = np.array(y0)

y2 = x0/(a*x0+b)

r2 = r2_score(y0, y2)    

plt.scatter(x0,y0,s=30,marker='o')

plt.xlim((0, 250))
plt.ylim((0, 1))

plt.plot(x1, y1, "blue")  
plt.title('account_attendee against percentile',fontsize=13) 
plt.xlabel('account_attendee',fontsize=12)  
plt.ylabel('percentile',fontsize=12)  

plt.grid(True, linestyle = "--", color = "g", linewidth = "0.5")
# plt.show()


p = round(9*b/(1-0.8*a),2)
# p = b/(math.log(0.8/a))
p =  round(p, 6)
# plt.scatter(p,0.8,s=20,marker='x')
# plt.vlines(p, 0, 0.8, colors = "c", linestyles = "dashed")
# plt.hlines(0.8, 0, p, colors = "c", linestyles = "dashed")
plt.text(p, 0.8, (float('%.6f'% p),0.8),ha='left', va='top', fontsize=11)
# 显示公式
m = round(max(y0)/10,1)
print(m)
plt.text(48, m, 'y= x/('+str(round(a,6))+'*x+'+str(round(b,6))+')', ha='right',fontsize=12)  
plt.text(48, m, r'$R^2=$'+str(round(r2,3)), ha='right', va='top',fontsize=12)

plt.show()