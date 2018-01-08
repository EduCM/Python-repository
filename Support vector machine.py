# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 09:24:58 2017

@author: if692068
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


#%%generar datos linealmente separables
np.random.seed(0)
x=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]] # el primer grupo de 20 datos tiene media -2 y el segundo 2
y=[0]*20+[1]*20

#%% crear y entrenar un model SVM
clf=svm.SVC(kernel='linear')
# clf=svm.SVC(kernel='poly',degree=2) # para cambiarlo a polinomeal
# clf=svm.SVC(kernel='rbf') # para cambiarlo a gausiano
clf.fit(x,y)

#%% obtener frontera
Wsvm=clf.coef_[0]
m= -Wsvm[0]/Wsvm[1]
b=-clf.intercept_[0]/Wsvm[1]

xx=np.linspace(-5,5)
yy=m*xx+b

VS=clf.support_vectors_[0]
yy_down=m*xx+(VS[1]-m*VS[0])
VS=clf.support_vectors_[-1]
yy_up=m*xx+(VS[1]-m*VS[0])
#%% dibujar puntos y frontera
plt.plot(xx,yy,'k-')
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolor='none')
plt.plot(xx,yy_down,'k--',xx,yy_up,'k--')
plt.axis('tight')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show