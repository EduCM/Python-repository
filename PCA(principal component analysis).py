# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 10:10:33 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

data=np.array([[2.5,2.4],[.5,.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,.9]])
#%%
plt.scatter(data[:,0],data[:,1])
plt.grid()
plt.show()

#%% tratar los datos antes del PCA

medias=data.mean(axis=0)
data_m=data-medias

plt.scatter(data_m[:,0],data_m[:,1])
plt.grid()
plt.show()

#%% matriz covarianza

data_cov=np.cov(data_m.transpose())

#%% calcular valores y vectores propios

w,v=np.linalg.eig(data_cov)
#%% dibujar las direcciones de los vectores propios

x=np.arange(-1.5,1.5,.1)
plt.scatter(data_m[:,0],data_m[:,1])
plt.plot(x,(v[1,0]/v[0,0])*x,'b--')
plt.plot(x,(v[1,1]/v[0,1])*x,'g--')
plt.axis('square')
plt.show()

#%% reordenar componentes
componentes=w[[1,0]]
transform=v[:,[1,0]]

#%% transformacion de los datos con los nuevos ejes

data_new=np.matrix(data_m)*np.matrix(transform)
#%% visualizar datos nuevos
plt.subplot(121)
plt.scatter(data_m[:,0],data_m[:,1])
plt.plot(x,(v[1,0]/v[0,0])*x,'b--')
plt.plot(x,(v[1,1]/v[0,1])*x,'g--')
plt.axis('square')

plt.subplot(122)
plt.scatter(data_new[:,0],data_new[:,1])
plt.hlines(0,-2,2,'g'),plt.vlines(0,-.4,.5,'b')
plt.grid()
plt.show()

#%% recuperar los datos originales a partir de la transformacion
data_r=np.matrix(data_new)*np.matrix(transform.transpose())+medias

#%%reduciendo la dimension de los datos

componentes=w[1]
transform=v[:,1]
transform=np.reshape(transform,[2,1])
data_new=np.matrix(data_m)*np.matrix(transform)
data_r=np.matrix(data_new)*np.matrix(transform.transpose())+medias

#%%
plt.scatter(data_r[:,0],data_r[:,0])
plt.show()

