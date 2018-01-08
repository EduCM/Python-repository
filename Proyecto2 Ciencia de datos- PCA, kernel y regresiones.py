# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:34:44 2017

@author: Edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn.metrics import(confusion_matrix,
                            precision_score,
                            recall_score,
                            f1_score,
                            accuracy_score)

#%%
accidente_file= 'creditcard.csv'
accidentes=pd.read_csv(accidente_file,
                       header=0,
                       sep=',',
                       parse_dates=False,
                       skip_blank_lines=True)
                       
#colnames=list(accidentes)

#%% análisis de componentes principales

medias=accidentes.mean(axis=0)
data=accidentes-medias
data_cov=np.cov(data.transpose())
w,v=np.linalg.eig(data_cov)

index=np.argsort(w)[::-1]   
componentes=w[index[0:20]]
transform=v[:,index[0:20]]          
                
data_new=np.matrix(data)*np.matrix(transform)
data_r=np.array(data_new)*np.matrix(transform.transpose())
medias=np.matrix(medias)
data_r=data_r+medias

resta_matrices=np.matrix(accidentes)-data_r

#%% REGRESIONES

#%% regresion con todas las variables
x1=accidentes.iloc[:,0:30]
y1=accidentes.iloc[:,30]  

xa1 = plf(degree=1).fit_transform(x1)
reglog1=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog1.fit(xa1,y1)# entrenar el modelo

yg1=reglog1.predict(xa1)
cm1=confusion_matrix(y1,yg1)
print ('\t Accuracy: %1.3f' %accuracy_score(y1,yg1))
print ('\t Precision: %1.3f' %precision_score(y1,yg1))
print ('\t Recall: %1.3f' %recall_score(y1,yg1))
print ('\t F1: %1.3f' %f1_score(y1,yg1))

w1=reglog1.coef_
plt.bar(np.arange(len(w1[0])),w1[0])
wabs1=np.abs(w1)
plt.bar(np.arange(len(wabs1[0])),wabs1[0])

umbral=0.05
index=wabs1>umbral
xas1=xa1[:,index[0]]

reglog1_2=linear_model.LogisticRegression(C=1)
reglog1_2.fit(xas1,y1)

yg1_2=reglog1_2.predict(xas1)
cm1_2=confusion_matrix(y1,yg1_2)
print ('\t Accuracy: %1.3f' %accuracy_score(y1,yg1_2))
print ('\t Precision: %1.3f' %precision_score(y1,yg1_2))
print ('\t Recall: %1.3f' %recall_score(y1,yg1_2))
print ('\t F1: %1.3f' %f1_score(y1,yg1_2))

#%% regresion con variables reducidas por componentes principales
x2=data_new[:,0:20]
y2=accidentes.iloc[:,30]    

xa2 = plf(degree=1).fit_transform(x2)
reglog2=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog2.fit(xa2,y2)# entrenar el modelo

yg2=reglog2.predict(xa2)
cm2=confusion_matrix(y2,yg2)
print ('\t Accuracy: %1.3f' %accuracy_score(y2,yg2))
print ('\t Precision: %1.3f' %precision_score(y2,yg2))
print ('\t Recall: %1.3f' %recall_score(y2,yg2))
print ('\t F1: %1.3f' %f1_score(y2,yg2))

w2=reglog2.coef_
plt.bar(np.arange(len(w2[0])),w2[0])
wabs2=np.abs(w2)
plt.bar(np.arange(len(wabs2[0])),wabs2[0])

umbral=0.1
index=wabs2>umbral
xas2=xa2[:,index[0]]

reglog2_2=linear_model.LogisticRegression(C=1)
reglog2_2.fit(xas2,y2)

yg2_2=reglog2_2.predict(xas2)
cm2_2=confusion_matrix(y2,yg2_2)
print ('\t Accuracy: %1.3f' %accuracy_score(y2,yg2_2))
print ('\t Precision: %1.3f' %precision_score(y2,yg2_2))
print ('\t Recall: %1.3f' %recall_score(y2,yg2_2))
print ('\t F1: %1.3f' %f1_score(y2,yg2_2))
#%%gráficas
indexx=np.array([9,10,11,12,13,14,15,16,17,20,30])
grado1 = pd.DataFrame({
                   "Accuracy":[.998,.999,.999,.999,.999,.999,.999,.999,.999,.999,.999,.999],
                   "Precision":[0,.777,.772,.816,.817,.847,.857,.861,.853,.849,.88,.839],
                   "Recall":[0,.411,.455,.522,.516,.561,.598,.606,.604,.606,.61,.604],
                   "F1":[0,.537,.573,.637,.633,.675,.704,.711,.707,.707,.72,.702]})
                 
grado1 = grado1.set_index(indexx)

plt.figure(figsize=(7,3))
plt.title('Accuracy 1er grado')
plt.ylabel('Accuracy level')
plt.xlabel('numero de variables')
plt.scatter(indexx,grado1.iloc[:,0])
plt.show()  

plt.figure(figsize=(7,3))
plt.title('Precision 1er grado')
plt.ylabel('Precision level')
plt.xlabel('numero de variables')
plt.scatter(indexx,grado1.iloc[1:12,2])
plt.show()                
                 
plt.figure(figsize=(7,3))
plt.title('F1 1er grado')
plt.ylabel('F1 level')
plt.xlabel('numero de variables')
plt.scatter(indexx,grado1.iloc[1:12,1])
plt.show()    

plt.figure(figsize=(7,3))
plt.title('Recall 1er grado')
plt.ylabel('Recall level')
plt.xlabel('numero de variables')
plt.scatter(indexx,grado1.iloc[1:12,3])
plt.show() 
                
indexx2=np.array([9,10,11,12,30])                
grado2 = pd.DataFrame({
                  "Accuracy":[.996,.997,.998,.996,.995],
                   "Precision":[.127,.156,.273,.127,.167],
                   "Recall":[.183,.211,.230,.236,.435],
                   "F1":[.15,.179,.249,.165,.242]})                
grado2 = grado2.set_index(indexx2)   

plt.figure(figsize=(7,3))
plt.title('Precision 2do grado')
plt.ylabel('Precision level')
plt.xlabel('numero de variables')
plt.scatter(indexx2,grado2.iloc[:,2])
plt.show()                
                 
plt.figure(figsize=(7,3))
plt.title('F1 2do grado')
plt.ylabel('F1 level')
plt.xlabel('numero de variables')
plt.scatter(indexx2,grado2.iloc[:,1])
plt.show()    

plt.figure(figsize=(7,3))
plt.title('Recall 2do grado')
plt.ylabel('Recall level')
plt.xlabel('numero de variables')
plt.scatter(indexx2,grado2.iloc[:,3])
plt.show()                 
                   