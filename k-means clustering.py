# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:24:26 2017

@author: Hector
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.cluster import hierarchy
import time
# esta seccion se crea con #%%
#%%
nsamples = 1500
random_state = 170
X, Y = make_blobs(n_samples = nsamples,
                  random_state = random_state)
#%%
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()
#%%
model = KMeans(n_clusters = 1,
               random_state = random_state,
               init = "random")
t0 = time.time()
model = model.fit(X)
t_model = time.time()-t0
Ypredict = model.predict(X)
t_model
##%%
plt.figure()
plt.scatter(X[:,0], X[:,1], c = Ypredict)
plt.show()  
model.inertia_ #promedio edl promedio de los centroides
model.cluster_centers_ #coordenadas de los centroides
#%%
inercia = np.zeros(15)
for k in np.arange(1,15):
    model = KMeans(n_clusters = k,
               random_state = random_state,
               init = "random")
    model = model.fit(X)
    inercia[k] = model.inertia_

plt.plot(np.arange(1,15), inercia[1:    ])
#%%
model = KMeans(n_clusters = 1,
               random_state = random_state,
               init = "random",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)
t0 = time.time()
model = model.fit(X)
t_model = time.time()-t0
Ypredict = model.predict(X)
t_model

#%% Para inicializar los centroides más alejados
model = KMeans(n_clusters = 1,
               random_state = random_state,
               init = "k-means++",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)
t0 = time.time()
model = model.fit(X)
t_model = time.time()-t0
Ypredict = model.predict(X)
t_model

#%% Pdesviacion estandar más grande para difuminar los clusters
model = KMeans(n_clusters = 10,
               random_state = random_state,
               init = "k-means++",
               max_iter = 300,
               n_init = 10,
               n_jobs= 1)

t0 = time.time()
model = model.fit(X)
t_model = time.time()-t0
Ypredict = model.predict(X)
t_model
#%%
inercia = np.zeros(15)
for k in np.arange(1,15):
    model = KMeans(n_clusters = k,
               random_state = random_state,
               init = "k-means++",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)

    model = model.fit(X)
    inercia[k] = model.inertia_

plt.plot(np.arange(1,15), inercia[1:])

aceleration = np.diff(inercia)
plt.plot(np.arange(2,15),-1*aceleration[1:])
plt.show()
k = aceleration.argmax()+2 #numero de cluster definidos
print ("clusters:" ,k)