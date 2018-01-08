# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:55:20 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThereshold
q
#%%
data=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]

#%%analisis varianza

xvar=np.var(data,axis=0)

#%%
sel=VarianceThereshold(thereshold=0.15)
xnew=sel.fit_transform(data)
