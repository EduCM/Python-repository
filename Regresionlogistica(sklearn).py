# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:28:10 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import scipy.optimize as opt
from sklearn import linear_model
from sklearn.metrics import(confusion_matrix,
                            precision_score,
                            recall_score,
                            f1_score,
                            accuracy_score)

#%%leer los datos

data=pd.read_csv('ex2data2.txt',header=None)

#%%separar datos de entrada y salida

x=data.iloc[:,0:2]
y=data.iloc[:,2]

#%% 
plt.scatter(x[0],x[1],c=y,edgecolor='w')

#%% preparar datos para la regresion
#xa=np.append(np.ones((len(y),1)),x.values,axis=1)   # para cambiarlo a culquier polinomio
ngrado=6
xa,coef=fun_polinomio(x,ngrado)
#%%definir el modelo logistico
reglog=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog.fit(xa,y)# entrenar el modelo

#%% dibujar frontera
#x1=np.arange(20,110,0.5)    #para datos 2
#x2=np.arange(20,110,0.5)  # para datos 2
x1=np.arange(-1,1.5,0.01)    
x2=np.arange(-1,1.5,0.01)  
X1,X2=np.meshgrid(x1,x2)

m,n=np.shape(X1)
X1r=np.reshape(X1,((m*n),1))
X2r=np.reshape(X2,((m*n),1))

#xm=np.append(np.ones((len(X1r),1)),X1r,axis=1)  # comentado para cambiarlo a culquier polinomio
#xm=np.append(xm,X2r,axis=1)     # para cambiarlo a culquier polinomio
xnew=pd.DataFrame(np.c_[X1r,X2r])
xm,coef= fun_polinomio(xnew,ngrado)
z=reglog.predict(xm)# reglog para dibujar la frontera
z=np.reshape(z,np.shape(X1))

#%%
plt.contour(X1,X2,z)
plt.scatter(x[0],x[1],c=y,edgecolor='w')
plt.show()
#%% reglog con datos originales
yg=reglog.predict(xa)
cm=confusion_matrix(y,yg)
print( '\t Accuracy: %1.3f' %accuracy_score(y,yg))
print ('\t Precision: %1.3f' %precision_score(y,yg))
print ('\t Recall: %1.3f' %recall_score(y,yg))
print ('\t F1: %1.3f' %f1_score(y,yg))


#%%
w=reglog.coef_
plt.bar(np.arange(len(w[0])),w[0])
#%% seleccion variables importantes 
wabs=np.abs(w)
plt.bar(np.arange(len(wabs[0])),wabs[0])

umbral=0.5
index=wabs>umbral
xas=xa[:,index[0]]

reglog1=linear_model.LogisticRegression(C=1)
reglog1.fit(xas,y)

yg=reglog1.predict(xas)
cm=confusion_matrix(y,yg)
print "\t Accuracy: %1.3f" %accuracy_score(y,yg)
print "\t Precision: %1.3f" %precision_score(y,yg)
print "\t Recall: %1.3f" %recall_score(y,yg)
print "\t F1: %1.3f" %f1_score(y,yg)

coefnew=coef[:,index[0]]