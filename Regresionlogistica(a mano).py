# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:56:30 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

#%%
data=pd.read_csv('ex2data1.txt',header=None)

#%%
x=data.iloc[:,0:2]
y=data.iloc[:,2]

#%%

plt.scatter(x[0],x[1],c=y,edgecolor='w')

#%% funcion logistica

def fun_log(v):
    return 1/(1+np.exp(-v))
    
#%% modelologistico
    
def reg_log(w,x,y):
    v=np.matrix(x)*np.matrix(w).transpose()
    return np.array(fun_log(v))[:,0]

#%% funcion de costo "J"

def fun_cost(w,x,y):
    yg=reg_log(w,x,y)
    return np.sum(-y*np.log(yg)-(1-y)*np.log(1-yg))/len(y)

#%% gradiente dj/d

def grad_cost(w,x,y):
    yg=reg_log(w,x,y)
    E=yg-y
    return np.array(np.matrix(E)* np.matrix(x)/len(y))[:,0]
    
#%% preparar los datos para la regresion
    
xa=np.append(np.ones((len(y),1)),x.values,axis=1)    
m,n=np.shape(xa)
w=np.zeros(n)

#%%encontrar solucion minima de la funcion de costo
args=(xa,y)
w0=np.zeros(n)
w_opt=opt.fmin(fun_cost,w0,args=args)
    
#%%simulacion de mi regresion logistica
yg=reg_log(w_opt,xa,y)
yg=np.round(yg)

#%%
x1=np.arange(20,110,0.5)    
x2=np.arange(20,110,0.5)  
X1,X2=np.meshgrid(x1,x2)

m,n=np.shape(X1)
X1r=np.reshape(X1,((m*n),1))
X2r=np.reshape(X2,((m*n),1))

xa=np.append(np.ones((len(X1r),1)),X1r,axis=1)    
xa=np.append(xa,X2r,axis=1)

yg=reg_log(w_opt,xa,y)
z=np.reshape(yg,(m,n))
z=np.round(z)

#%%
plt.contour(X1,X2,z)
plt.scatter(x[0],x[1],c=y,edgecolor='w')
plt.show()