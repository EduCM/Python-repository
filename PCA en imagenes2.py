# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 10:38:19 2017

@author: if692068
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import datasets

#%%importar datos sklearn
digits=datasets.load_digits()

#%% visualizar digitos
n_dig=10
for k in np.arange(n_dig):
    plt.subplot(2,n_dig/2,k+1)
    plt.axis('off')
    plt.imshow(digits.images[k],cmap=plt.cm.gray_r)
    plt.title('Digit: %i' % k)

#%% reordenar valores de pixeles

labels=digits.target[0:100]
data=digits.data[0:100,:]

#%% restarle la media a los datos
media=data.mean(axis=0)
datam=data-media

data_cov=np.cov(datam.transpose())
w,v=np.linalg.eig(data_cov)

#%% seleccionando componentes principales

index=np.argsort(w)[::-1]
componentes=w[index[0:1]]
transform=v[:,index[0:1]]

#%%proyectando la imagen en los nuevos ejes

data_new=np.matrix(datam)*np.matrix(transform)
#%%visualizar proyecciones de los digitos
plt.scatter(data_new,data_new*0,c=labels)

#%%
index=np.argsort(w)[::-1]
componentes=w[index[0:2]]
transform=v[:,index[0:2]]

data_new=np.matrix(datam)*np.matrix(transform)

plt.scatter(data_new[:,0],data_new[:,1],c=labels)
plt.show()

#obetener porcentajr de importancia de los componentes

porcentaje=w/np.sum(w)
porcentaje_acum=np.cumsum(porcentaje)

plt.bar(np.arange(len(porcentaje)),porcentaje)
plt.show()
plt.bar(np.arange(len(porcentaje_acum)),porcentaje_acum)
plt.show()
