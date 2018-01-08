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
dk=hierarchy.dendrogram(k)