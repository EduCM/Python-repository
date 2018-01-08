# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:02:26 2017

@author: Edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% leer archivos

#CASO A
data_a='dataset_3a.csv'
dataA=pd.read_csv(data_a,header=None,sep=',') # no tiene titulos
dataA=dataA.iloc[0:2000,0:3]
dataA.columns=['V1','V2','Class']

#CASO B
data_b='dataset_3b.csv'
dataB=pd.read_csv(data_b,header=None,sep=',') # no tiene titulos
dataB=dataB.iloc[0:2000,0:3]
dataB.columns=['V1','V2','Class']

#%% gráficos

#CASO A
index=dataA['Class']==1
dataA_1=dataA.loc[index,:]
index=dataA['Class']==0
dataA_2=dataA.loc[index,:]
plt.figure()
plt.title('Caso A')
plt.ylabel('V2')
plt.xlabel('v1')
plt.scatter(dataA_2.V1,dataA_2.V2,color='m',s=10)
plt.scatter(dataA_1.V1,dataA_1.V2,color='y',s=10)
plt.show() 

#CASO B
index=dataB['Class']==1
dataB_1=dataB.loc[index,:]
index=dataB['Class']==0
dataB_2=dataB.loc[index,:]
plt.figure()
plt.title('Caso B')
plt.ylabel('V2')
plt.xlabel('v1')
plt.scatter(dataB_2.V1,dataB_2.V2,color='m',s=4)
plt.scatter(dataB_1.V1,dataB_1.V2,color='y',s=4)
plt.show()  

#%% análisis de componentes principales PCA

#PAra CASO A
dataAA=dataA.iloc[:,0:2]
medias=dataAA.mean(axis=0)
data1=dataAA-medias
data_cov1=np.cov(data1.transpose())
w1,v1=np.linalg.eig(data_cov1)
num_variables=1
perdida_info=1-(sum(w1[0:num_variables])/sum(w1))
index=np.argsort(w1)[::-1]   
componentes1=w1[index[0:num_variables]]
transform1=v1[:,index[0:num_variables]]          
                
data_new1=np.matrix(data1)*np.matrix(transform1)
data_r1=np.array(data_new1)*np.matrix(transform1.transpose())
medias=np.matrix(medias)
data_r1=data_r1+medias

resta_matrices=np.matrix(dataAA)-data_r1
suma=sum(np.absolute(resta_matrices))


#PAra CASO B
dataBB=dataB.iloc[:,0:2]
medias=dataBB.mean(axis=0)
data2=dataBB-medias
data_cov2=np.cov(data2.transpose())
w2,v2=np.linalg.eig(data_cov2)
num_variables=1
index=np.argsort(w2)[::-1]   
componentes2=w2[index[0:num_variables]]
transform2=v2[:,index[0:num_variables]]   
perdida_info=1-(sum(componentes2[0:num_variables])/sum(w2))       
                
data_new2=np.matrix(data2)*np.matrix(transform2)
data_r2=np.array(data_new2)*np.matrix(transform2.transpose())
medias=np.matrix(medias)
data_r2=data_r2+medias

resta_matrices=np.matrix(dataBB)-data_r2
suma=sum(np.absolute(resta_matrices))

#%% gráficos datos después de PCA

#CASO A
data_new1=pd.DataFrame(data_new1)
data_new1.columns=['V']
dataA_PCA=data_new1.join(dataA.Class)
index=dataA_PCA['Class']==1
dataA_1=dataA_PCA.loc[index,:]
index=dataA_PCA['Class']==0
dataA_2=dataA_PCA.loc[index,:]
plt.figure()
plt.title('Caso A después de PCA')
plt.ylabel('V')
x=np.arange(0,1000)
plt.scatter(x,dataA_2.V,color='m',s=10)
plt.scatter(x,dataA_1.V,color='y',s=10)
plt.show() 

#CASO B
data_new2=pd.DataFrame(data_new2)
data_new2.columns=['V']
dataB_PCA=data_new2.join(dataB.Class)
index=dataB_PCA['Class']==1
dataB_1=dataB_PCA.loc[index,:]
index=dataB_PCA['Class']==0
dataB_2=dataB_PCA.loc[index,:]
plt.figure()
plt.title('Caso B después de PCA')
plt.ylabel('V')
x1=np.arange(0,1333)
x2=np.arange(0,667)
plt.scatter(x1,dataB_2.V,color='m',s=10)
plt.scatter(x2,dataB_1.V,color='y',s=10)
plt.show()  