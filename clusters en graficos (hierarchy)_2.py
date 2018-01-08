# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:44:54 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime
import scipy.cluster.hierarchy as hc
#%%
data= DataReader('GrumaB.MX','yahoo',datetime(2016,4,2),datetime(2017,4,2))
#%%visualizar
plt.subplot(211)
plt.plot(data['High'],'r--')
plt.plot(data['Low'],'r--')
plt.plot(data['Close'],'b--')
plt.xlabel('Time')
plt.ylabel('Precio')
plt.grid()

plt.subplot(212)
plt.plot(data['Volume'],'K')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.grid()
plt.show()

#%%
dat=data['Adj Close']
nv=5
nprices=len(dat)
dat_new=np.zeros((nprices-nv,nv))
for k in np.arange(nv):
    dat_new[:,k]=dat[k:nprices-nv+k]
    
#%% 
    
plt.subplot(211)
plt.plot(dat_new)
plt.xlabel('time')
plt.ylabel('price')

plt.subplot(212)
plt.plot(dat_new.transpose())
plt.xlabel('time')
plt.ylabel('price')
#%%normalizar datos
variabletemporal=dat_new.transpose()
dat_new=np.transpose((variabletemporal-variabletemporal.mean(axis=0))/variabletemporal.std(axis=0))

#%% clustering con HC
z=hc.linkage(dat_new,'ward')
plt.figure()
hc.dendrogram(z)
plt.show()
#%% generar los clusters
max_d=10
clusters=hc.fcluster(z,max_d,criterion='distance')

#%%visualizar un cluster

nclus=2
index=clusters==nclus
datclust=dat_new[index,]
mdatclust=datclust.mean(axis=0)
plt.subplot(211)
plt.plot(datclust.transpose())
plt.xlabel('interval')
plt.ylabel('price')

plt.subplot(212)
plt.plot(mdatclust)
plt.xlabel('interval')
plt.ylabel('price')
plt.show()

#%% visualizar todos los patrones promedio 

n_subfig=np.ceil(np.sqrt(len(np.unique(clusters))))
for k in np.arange(1,len(np.unique(clusters))+1):
    datclust=dat_new[clusters==k,]
    plt.subplot(n_subfig,n_subfig,k)
    plt.plot(datclust.mean(axis=0))
    plt.ylabel('cluster %d'%k)
    
plt.show()

#%% graficar en el tiempo la aparicion de los patrones encontrados 
plt.subplot(211)
plt.plot(dat)
plt.xlabel('time')
plt.ylabel('precios')

plt.subplot(212)
plt.bar(np.arange(nv,len(dat)),clusters)
plt.show()

#%%reordenar los patrones para ver las secuencias
n_pat=len(clusters)
sec_clust=np.zeros((n_pat-1,2))
sec_clust[:,0]=clusters[0:n_pat-1]
sec_clust[:,1]=clusters[1:n_pat]

#%% calcular la matriz de transicion
nclusters=np.unique(clusters)
m=np.zeros((len(nclusters),len(nclusters)))
for k in np.arange(1,len(nclusters)+1):
    for l in np.arange(1,len(nclusters)+1):
        index=(sec_clust[:,0]==k)&(sec_clust[:,1]==l)
        m[l-1,k-1]=np.sum(index)
m=m/m.sum(axis=0)
