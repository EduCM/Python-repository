# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:10:07 2017

@author: Edu
"""
### ejercicio 1 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
#%%
data_file='dataset_1.csv'
data=pd.read_csv(data_file,header=0)

#%% hierarchical clustering 

k=hierarchy.linkage(data,'complete')
plt.figure(figsize=(25,10))
plt.title('dendograma completo')
plt.ylabel('distancia')
plt.xlabel('indice de la muestra')
dk=hierarchy.dendrogram(k)
plt.show()

#%% gráfica de 'codo'
last=k[-10:,2]
last_rev=last[::-1]
idxs=np.arange(1,len(last_rev)+1)
plt.plot(idxs,last_rev)

aceleracion=np.diff(last)
aceleracion_rev=aceleracion[::-1]
plt.plot(idxs[:-1]+1,aceleracion_rev)
plt.show()

#%% algoritmo kmeans

inercia = np.zeros(15)
for k in np.arange(1,15):
    model = KMeans(n_clusters = k,
               init = "random",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)

    model = model.fit(data)
    inercia[k] = model.inertia_

plt.plot(np.arange(1,15), inercia[1:])

aceleration = np.diff(inercia)
plt.plot(np.arange(2,15),-1*aceleration[1:])
plt.show()