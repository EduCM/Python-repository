# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 09:16:33 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import datasets

#%%lectura imagen
img=mpimg.imread('images.png')
imshow=plt.imshow(img)

#%% reordenar valores de pixeles

d=img.shape
img_rshape=np.reshape(img,(d[0]*d[1],d[2])) # cada fila es un pixel_(cada pixel tiene 4 dimensiones de colores)

#%% restarle la media a los datos
data=img_rshape
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
#%% recuperar la imagen

data_mr=np.matrix(data_new)*np.matrix(transform.transpose())+media

img_r=img.copy()
img_r[:,:,0]=data_mr[:,0].reshape((d[0],d[1]))
img_r[:,:,1]=data_mr[:,1].reshape((d[0],d[1]))
img_r[:,:,2]=data_mr[:,2].reshape((d[0],d[1]))
img_r[:,:,3]=data_mr[:,3].reshape((d[0],d[1]))

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_r)
