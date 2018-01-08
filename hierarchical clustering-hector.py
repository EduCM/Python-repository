# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:15:45 2017

@author: Hector
"""


#paqueterias
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as sc
import sklearn
from scipy.cluster import hierarchy
# esta seccion se crea con #%%
#%%
accidents_file = "C:/Users/Hector/Desktop/CDIN 2017/Data/Accidents_2015.csv"
accidents = pd.read_csv(accidents_file, 
                        header=0,
                        sep=',',
                        index_col=0,
                        parse_dates=False,
                        nrows=150000,
                        skip_blank_lines=True)
#%%
minidata = accidents.iloc[0:1000,[2,3,6,7,5,9]]
#%%
minidata_dummy = minidata.iloc[:,0:4]
temp = pd.get_dummies(minidata["Accident_Severity"])
temp.columns = ["Severity_1","Severity_2","Severity_3"]
minidata_dummy = minidata_dummy.join(temp)

temp = pd.get_dummies(minidata["Day_of_Week"])
temp.columns = ["Sunday","Monday","tuesday","Wednesday","thursday","friday","Saturday"]
minidata_dummy = minidata_dummy\
.join(temp)
#%%
minidata_dummy = (minidata_dummy-minidata_dummy.mean(axis=0))/\
minidata_dummy.std(axis=0)
#%%
Z = hierarchy.linkage(minidata_dummy,"complete","euclidean")
plt.figure(figsize=(10,10))
dn = hierarchy.dendrogram(Z)

#%%
plt.title("Dendograma Truncado")
plt.ylabel("Distancia")
plt.xlabel("Indice de la muestra")
hierarchy.dendrogram(Z,
                     truncate_mode="lastp",
                     p=12,
                     show_leaf_counts = True,
                     show_contracted = True,
                     leaf_font_size = 30,
                     leaf_rotation = 295)
plt.axhline(10,c="k")
plt.show()
#%% no supervisado
#inconcistencia de grupo es la distancia promedio entre grupos
last = Z[:,2]
last_rev = last[::-1]
idxs = np.arange(1,len(last_rev)+1)
plt.plot(idxs,last_rev)

aceleration = np.diff(last)
aceleration_rev = aceleration[::-1]
plt.plot(idxs[:-1]+1,aceleration_rev)
plt.show()
k = aceleration_rev.argmax()+2 #numero de cluster definidos
print ("clusters:" ,k)

#%%como saber que dato pertenece a que cluster
k = 10
index = hierarchy.fcluster(Z,k,criterion="distance")




































