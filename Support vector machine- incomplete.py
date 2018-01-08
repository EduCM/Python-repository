# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 09:24:58 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

#%%
iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

#%% crear y entrenar un model SVM
svc=svm.SVC(kernel='linear').fit(x,y)
scv_poly=svm.SVC(kernel='poly',degree=3).fit(x,y)
svc_rbf=svm.SVC(kernel='rbf').fit(x,y)

#%% dibujar puntos y frontera
plt.scatter(x[:,0],x[:,1],c=y)
#plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolor='none')
#plt.plot(xx,yy_down,'k--',xx,yy_up,'k--')
plt.axis('tight')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show