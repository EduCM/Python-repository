# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:43:27 2017

@author: Edu
"""
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn.metrics import(confusion_matrix,
                            precision_score,
                            recall_score,
                            f1_score,
                            accuracy_score)
#%%
data_file='dataset_2.csv'
data=pd.read_csv(data_file,header=0,skip_blank_lines=True,parse_dates=False)
data=data.iloc[0:2000,0:3]

quick_report_obj=data.describe(include=['object']).transpose()

data2=data.iloc[:,0:2]
temp=pd.get_dummies(data['Class'])   
data2=data2.join(temp.Regular) # 1= Regular, 0=Malo
#%% separar datos de de entranamiento y prueba
porcentaje_datos_m=.66666
ind=round(porcentaje_datos_m*len(data2.V1))
data_m=data2.iloc[0:ind,:]
data_p=data2.iloc[ind:len(data2.V1),:]

#%% entrenar regresion con datos de entrenamiento
x=data_m.iloc[:,0:2]
y=data_m.iloc[:,2:3]    

xa= plf(degree=2).fit_transform(x)
reglog=linear_model.LogisticRegression(C=1)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog.fit(xa,y)# entrenar el modelo

yg=reglog.predict(xa)
cm=confusion_matrix(y,yg)
print ('\t Accuracy: %1.3f' %accuracy_score(y,yg))
print ('\t Precision: %1.3f' %precision_score(y,yg))
print ('\t Recall: %1.3f' %recall_score(y,yg))
print ('\t F1: %1.3f' %f1_score(y,yg))

#%% hacer pronósticos y comparar con datos de prueba
x1=data_p.iloc[:,0:2]
y1=data_p.iloc[:,2:3]    

xa1= plf(degree=2).fit_transform(x1)

yg1=reglog.predict(xa1)
cm=confusion_matrix(y1,yg1)
print ('\t Accuracy: %1.3f' %accuracy_score(y1,yg1))
print ('\t Precision: %1.3f' %precision_score(y1,yg1))
print ('\t Recall: %1.3f' %recall_score(y1,yg1))
print ('\t F1: %1.3f' %f1_score(y1,yg1))


