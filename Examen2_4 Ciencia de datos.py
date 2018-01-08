# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:25:20 2017

@author: Edu
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn.metrics import(confusion_matrix,
                            precision_score,
                            recall_score,
                            f1_score,
                            accuracy_score)
#%%
data_file='dataset_4.csv'
data=pd.read_csv(data_file,header=None)
data=data.iloc[0:3000,0:3]
data.columns=['V1','V2','Class']
unique_values_counts=pd.DataFrame(columns=['unique values'])
for v in list(data.columns.values):
    unique_values_counts.loc[v]=[data[v].nunique()]   
 
#%% graficar datos

index=data['Class']==1
data_1=data.loc[index,:]
index=data['Class']==0
data_0=data.loc[index,:]
index=data['Class']==2
data_2=data.loc[index,:]

plt.figure()
plt.title('DATOS')
plt.ylabel('V2')
plt.xlabel('v1')
plt.scatter(data_2.V1,data_2.V2,color='m',s=10)
plt.scatter(data_1.V1,data_1.V2,color='y',s=10)
plt.scatter(data_0.V1,data_0.V2,color='b',s=10)
plt.show() 
   
#%% regresión logística one VS all
    
# primera regresión 1=1, 0=0, 2=0  ############################
index=data['Class']!=1
data_1=data.iloc[:,2]
data_1[index]=0

x=data.iloc[:,0:2]
y=data_1 
y=np.ravel(y)

xa=plf(degree=2).fit_transform(x)
reglog=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog.fit(xa,y)# entrenar el modelo

yg=reglog.predict(xa)
cm=confusion_matrix(y,yg)
print ('\t Accuracy: %1.3f' %accuracy_score(y,yg))
print ('\t Precision: %1.3f' %precision_score(y,yg))
print ('\t Recall: %1.3f' %recall_score(y,yg))
print ('\t F1: %1.3f' %f1_score(y,yg))

# Segunda regresión 1=0, 0=1, 2=0 ############################

data=pd.read_csv(data_file,header=None)
data=data.iloc[0:3000,0:3]
data.columns=['V1','V2','Class']

index=data['Class']!=0
index2=data['Class']==0
data_2=data.iloc[:,2]
data_2[index2]=1
data_2[index]=0

x2=data.iloc[:,0:2]
y2=data_2 
y2=np.ravel(y2)

xa2=plf(degree=2).fit_transform(x2)
reglog2=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog2.fit(xa2,y2)# entrenar el modelo

yg2=reglog2.predict(xa2)
cm2=confusion_matrix(y2,yg2)
print ('\t Accuracy: %1.3f' %accuracy_score(y2,yg2))
print ('\t Precision: %1.3f' %precision_score(y2,yg2))
print ('\t Recall: %1.3f' %recall_score(y2,yg2))
print ('\t F1: %1.3f' %f1_score(y2,yg2))

# tercera regresión 1=0, 0=0, 2=1 ############################

data=pd.read_csv(data_file,header=None)
data=data.iloc[0:3000,0:3]
data.columns=['V1','V2','Class']

index=data['Class']!=2
index2=data['Class']==2
data_3=data.iloc[:,2]
data_3[index2]=1
data_3[index]=0

x3=data.iloc[:,0:2]
y3=data_3 
y3=np.ravel(y3)

xa3=plf(degree=4).fit_transform(x3)
reglog3=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog3.fit(xa3,y3)# entrenar el modelo
    
yg3=reglog3.predict(xa3)
cm3=confusion_matrix(y3,yg3)
print ('\t Accuracy: %1.3f' %accuracy_score(y3,yg3))
print ('\t Precision: %1.3f' %precision_score(y3,yg3))
print ('\t Recall: %1.3f' %recall_score(y3,yg3))
print ('\t F1: %1.3f' %f1_score(y3,yg3))                   

#%% Vector Soporte  SVM 
X = data.iloc[:,:2]
Y= data.iloc[:,2]

# crear y entrenar modelo svm
svc = svm.SVC(kernel='linear').fit(X,Y)
svc_poly = svm.SVC(kernel='poly',degree=4).fit(X,Y) 
svc_rbf = svm.SVC(kernel='rbf').fit(X,Y)

# Crear un mesh para dibujar la frontera
h = 0.02
xx,yy = np.meshgrid(np.arange(4,8,h),np.arange(1.5,4.5,h))

# Dibujar las fronteras
titles = ['SVM Lineal', 'SVM Polinomial', 'SVM Radial']
for j,clf in enumerate((svc,svc_poly,svc_rbf)):
    plt.subplot(2,2,j+1)
    plt.subplots_adjust(wspace=0.4,hspace=0.6)
    P = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    P = P.reshape(xx.shape)
    plt.contourf(xx,yy,P,cmap=plt.cm.Paired,alpha =0.7)
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=Y,)
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.title(titles[j])
    
    Yg =clf.predict(X)
    cm = confusion_matrix(Y,Yg)
    print(titles[j])
    print('\t Accuracy: %1.3f' % accuracy_score(Y,Yg))
    print('\t Precision: %1.3f' % precision_score(Y,Yg))
    print('\t Recall: %1.3f' % recall_score(Y,Yg))
    print('\t F1: %1.3f' % f1_score(Y,Yg))
    
plt.show()