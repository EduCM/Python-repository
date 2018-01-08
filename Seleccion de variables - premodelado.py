# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:12:55 2017

@author: Edu
"""

import numpy as np
import pandas as pd

#%% DATOS
accidente_file= 'Accidents_2015.csv'
accidentes=pd.read_csv(accidente_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)

minidata=accidentes.iloc[0:50,[2,3,6,7,5,9]]
minidata_dummy=minidata.iloc[:,0:4]
temp=pd.get_dummies(minidata['Accident_Severity'])   
temp.columns=['Severity_2','Severity_3']

minidata_dummy=minidata_dummy.join(temp)
temp=pd.get_dummies(minidata['Day_of_Week'])   
temp.columns=['Sunday','Monday','Tuesday','wednesday','Thursday','Friday','saturday']
minidata_dummy=minidata_dummy.join(temp)
                       
#%% analisis de varianza
xvar=np.var(minidata_dummy,axis=0)  
minidata_dummy=minidata_dummy.iloc[0:50,[2,3,6,8,9,10,11,12]]
xvar=np.var(minidata_dummy,axis=0)

#%% análisis de correlación
correlacion=np.corrcoef(minidata_dummy.transpose())        

#%% análisis de componentes principales

medias=minidata_dummy.mean(axis=0)
minidata_dummy_m=minidata_dummy-medias
data_cov=np.cov(minidata_dummy_m.transpose())
w,v=np.linalg.eig(data_cov)

index=np.argsort(w)[::-1]   
componentes=w[index[0:9]]
transform=v[:,index[0:9]]          
                 
data_new=np.matrix(minidata_dummy_m)*np.matrix(transform)
data_r=np.array(data_new)*np.matrix(transform.transpose())
medias=np.matrix(medias)
data_r=data_r+medias

resta_matrices=np.matrix(minidata_dummy)-data_r

                 
                       