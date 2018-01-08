# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:29:00 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

#%%

data=np.array([662,255,412,996,295,468,268,400,754,564,138,219,869,669,500])

#%%
z=hierarchy.linkage(data,'complete')

#%%
plt.figure()
dn=hierarchy.dendrogram(z)

#%%
fig,axes=plt.subplots(1,2)
dn1=hierarchy.dendrogram(z,ax=axes[0],orientation='top')
dn2=hierarchy.dendrogram(z,ax=axes[1],orientation='right')

#%% clustering en dos dimensiones

np.random.seed(4711)
a=np.random.multivariate_normal([10,0],[[3,1],[1,4]],size=[100,])# 10 y 0 medias, 3 varianza x1 ,1 cov, 1 ,cov, 4 var x2
b=np.random.multivariate_normal([0,20],[[3,1],[1,4]],size=[50,])
x=np.concatenate((a,b),)
plt.scatter(x[:,0],x[:,1])
plt.show()

k=hierarchy.linkage(x,'complete')
plt.figure(figsize=(25,10))
plt.title('dendograma completo')
plt.ylabel('distancia')
plt.xlabel('indice de la muestra')
dk=hierarchy.dendrogram(k)
plt.show()

#%%
idx=[33,68,62];
plt.figure(figsize=(10,8))
plt.scatter(x[:,0],x[:,1])
plt.scatter(x[idx,0],x[idx,1],c='r')
plt.show()

#%%
plt.title('dendograma truncado')
plt.ylabel('distancia')
plt.xlabel('indice de la muestra')
hierarchy.dendrogram(k,truncate_mode='lastp',p=12,show_leaf_counts=False,show_contracted=True,leaf_font_size=12,leaf_rotation=45)
plt.show()

#%%
last=k[-10:,2]
last_rev=last[::-1]
idxs=np.arange(1,len(last_rev)+1)
plt.plot(idxs,last_rev)

aceleracion=np.diff(last)
aceleracion_rev=aceleracion[::-1]
plt.plot(idxs[:-1]+1,aceleracion_rev)
plt.show()

j=aceleracion_rev.argmax()+2
print("clusters:",j)

#%% como saber que datos pertenece a su respectivo cluster 
index=hierarchy.fcluster(k,j,criterion='maxclust')
