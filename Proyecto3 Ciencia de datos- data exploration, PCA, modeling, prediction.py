# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:53:05 2017

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

#%% leer archivo
data_file='HR_comma_sep.csv'
rh=pd.read_csv(data_file,header=0) # si tiene titulos
#%% nombres columnas
columns=pd.DataFrame(list(rh.columns.values),columns=['Col names'])
#%% Análisis de datos
data_type=pd.DataFrame(rh.dtypes,columns=['Data type'])

missing_data_counts=pd.DataFrame(rh.isnull().sum(),columns=['missing values']) 

present_data_couts=pd.DataFrame(rh.count(),columns=['present values'])

unique_values_counts=pd.DataFrame(columns=['unique values'])
for v in list(rh.columns.values):
    unique_values_counts.loc[v]=[rh[v].nunique()] 
    
minimum_values=pd.DataFrame(columns=['minimum values'])         
for v in list(rh.columns.values):
   try: 
    minimum_values.loc[v]=[rh[v].min()]    
   except:
    pass  

max_values=pd.DataFrame(columns=['maximun values'])         
for v in list(rh.columns.values):
    max_values.loc[v]=[rh[v].max()]  
    
data_quality_report=data_type.join(missing_data_counts).join(present_data_couts).join(unique_values_counts).join(minimum_values).join(max_values)    

quick_report_obj=rh.describe(include=['object']).transpose()

conteo_renuncias=float(rh.left.sum(axis=0))
porct_renuncias=conteo_renuncias/(len(rh.satisfaction_level))

 #%%DUMMYES
rh_2=rh.iloc[:,0:8]
temp=pd.get_dummies(rh['sales'])
temp1=pd.get_dummies(rh['salary'])   
data=rh_2.join(temp1).join(temp)

#%% cambiar orden de las filas randomente
data=data.sample(frac=1)
#%%separar datos muestra y datos prueba
porcentaje_datos_m=.66666
ind=round(porcentaje_datos_m*len(data.left))
ind=int(ind)
data_m=data.iloc[0:ind,:]
data_p=data.iloc[ind:len(rh .left),:]
conteo_renuncias_m=float(data_m.left.sum(axis=0))
porct_renuncias_m=conteo_renuncias_m/len(data_m.satisfaction_level)
conteo_renuncias_p=float(data_p.left.sum(axis=0))
porct_renuncias_p=conteo_renuncias_p/len(data_p.satisfaction_level)
data_m_r=data_m.iloc[:,6]
data_m=data_m.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
data_p_r=data_p.iloc[:,6]
data_p=data_p.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
#%% nivel satisfaccion
data_satis=data.iloc[:,[0,6]]
kk=np.histogram(data_satis)
index=data_satis['left']==1
data_satis_1=data_satis.loc[index,:]
index=data_satis['left']==0
data_satis_0=data_satis.loc[index,:]
hist_0=plt.hist(data_satis_0.satisfaction_level)
hist_1=plt.hist(data_satis_1.satisfaction_level)

#%% análisis de componentes principales(seleccion pre-modelado)

medias=data_m.mean(axis=0)
data=data_m-medias
data_cov=np.cov(data.transpose())
w,v=np.linalg.eig(data_cov)
num_variables=10
index=np.argsort(w)[::-1]   
componentes=w[index[0:num_variables]]
transform=v[:,index[0:num_variables]]          
perdida_info=1-(sum(componentes[0:num_variables])/sum(w))
               
data_new=np.matrix(data)*np.matrix(transform)
data_r=np.array(data_new)*np.matrix(transform.transpose())
medias=np.matrix(medias)
data_r=data_r+medias

resta_matrices=np.matrix(data_m)-data_r
suma=sum(resta_matrices)
#%% normilizar datos después del PCA
data_new=pd.DataFrame(data_new)  
data_new=data_new.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
data_new=np.matrix(data_new)
#%% entrenar regresion con variables reducidas por componentes principales
x2=data_new[:,0:num_variables]
y2=data_m_r

xa2 = plf(degree=3).fit_transform(x2)
reglog2=linear_model.LogisticRegression(C=.9)#crear el modelo, # la "c" es la c de la regresion regularizada
reglog2.fit(xa2,y2)# entrenar el modelo

yg2=reglog2.predict(xa2)
cm2=confusion_matrix(y2,yg2)
print ('\t Accuracy: %1.3f' %accuracy_score(y2,yg2))
print ('\t Precision: %1.3f' %precision_score(y2,yg2))
print ('\t Recall: %1.3f' %recall_score(y2,yg2))
print ('\t F1: %1.3f' %f1_score(y2,yg2))

# seleccion post-modelado

w1=reglog2.coef_
wabs1=np.abs(w1)
plt.bar(np.arange(len(wabs1[0])),wabs1[0])

umbral=0.01
index=wabs1>umbral
xas2=xa2[:,index[0]]

reglog2_2=linear_model.LogisticRegression(C=1)
reglog2_2.fit(xas2,y2)

yg2_2=reglog2_2.predict(xas2)
cm2_2=confusion_matrix(y2,yg2_2)
print ('\t Accuracy: %1.3f' %accuracy_score(y2,yg2_2))
print ('\t Precision: %1.3f' %precision_score(y2,yg2_2))
print ('\t Recall: %1.3f' %recall_score(y2,yg2_2))
print ('\t F1: %1.3f' %f1_score(y2,yg2_2))
#%% ACP y predicción con la segunda muestra de los datos

# ACP
mediasp=data_p.mean(axis=0)
datap=data_p-mediasp
data_covp=np.cov(datap.transpose())
wp,vp=np.linalg.eig(data_covp)
perdida_info=1-(sum(wp[0:num_variables])/sum(wp))
index=np.argsort(wp)[::-1]   
componentesp=wp[index[0:num_variables]]
transformp=vp[:,index[0:num_variables]]          
                
data_newp=np.matrix(datap)*np.matrix(transformp)
data_rp=np.array(data_newp)*np.matrix(transformp.transpose())
mediasp=np.matrix(mediasp)
data_rp=data_rp+mediasp

resta_matricesp=np.matrix(data_p)-data_rp
sumap=sum(resta_matricesp)

## normalizar datos despues ACP
data_newp=pd.DataFrame(data_newp)  
data_newp=data_newp.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
data_newp=np.matrix(data_newp)

#Regresion

x3=data_newp[:,0:num_variables]
y3=data_p_r 

xa3 = plf(degree=3).fit_transform(x3)

umbral=0.01
index=wabs1>umbral
xa3=xa3[:,index[0]]

yg3=reglog2_2.predict(xa3)
cm3=confusion_matrix(y3,yg3)
print ('\t Accuracy: %1.3f' %accuracy_score(y3,yg3))
print ('\t Precision: %1.3f' %precision_score(y3,yg3))
print ('\t Recall: %1.3f' %recall_score(y3,yg3))
print ('\t F1: %1.3f' %f1_score(y3,yg3))

