# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:46:09 2017

@author: if698591
Armando Díaz González
Sergio Palafox Pucheta

"""

import numpy as np
import pandas as pd
#%%
def fun_polinomio(X,ngrado):
    Xp=np.ones((np.size(X,0),1))
    nvar=np.size(X,1)
    coef=np.zeros((1,nvar))
    for g in np.arange(1,ngrado+1):
    #Obtener la tabla de coeficientes
        Atemp=np.concatenate((np.array(np.arange(g-1,0,-1)).reshape(-1,1),np.array(np.arange(1,g)).reshape(-1,1)),axis=1)
        A=-1*np.ones((1,nvar))
        for i in np.arange(0,nvar-1):
            for j in np.arange(i+1,nvar):
                Btemp=np.zeros((g-1,nvar))
                Btemp[:,i]=Atemp[:,0]
                Btemp[:,j]=Atemp[:,1]
                A=np.concatenate((A,Btemp),axis=0)
        A = np.concatenate((A,g*np.identity(nvar)))
        A = A[1:]
    # elevar al grado ngrado
        for k in np.arange(1,np.size(A,0)+1):
            temp=np.ones((np.size(X,0),1))
            for j in np.arange(1,nvar+1):
                temp=np.multiply(temp,np.matrix((X.iloc[:,j-1]**A[k-1,j-1])).transpose())
            Xp=np.concatenate((Xp,temp),axis=1)
        coef=np.concatenate((coef,A))
    coef=np.transpose(coef)
    return (Xp,coef)