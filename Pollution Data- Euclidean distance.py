# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:54:22 2017

@author: if692068
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
#%%
data_file='Datos_2015.csv'
pollution=pd.read_csv(data_file,header=0) # si tiene titulos
#%%
minidata=pollution.iloc[0:10,2:7]

#%%
minidata_norms=(minidata-minidata.mean(axis=0))/minidata.std(axis=0)   

#%%
plt.scatter(minidata_norms.iloc[:,0],minidata_norms.iloc[:,3])
plt.xlabel('CO')
plt.ylabel('PM10')
plt.axis('square') # para que los dos ejes tengan la misma escala
plt.show()

#%%
dist1=sc.squareform(sc.pdist(minidata_norms,'euclidean')) #distancias entre filas, una hora comparada contra otra
dist2=sc.squareform(sc.pdist(minidata_norms.transpose(),'euclidean')) #distancia entre columnas 

#%%
dist3=sc.squareform(sc.pdist(minidata_norms,'cosine')) #angulo entre ellos
dist4=sc.squareform(sc.pdist(minidata_norms.transpose(),'cosine'))