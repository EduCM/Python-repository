# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:08:26 2017

@author: Edu
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as plf
from sklearn.metrics import r2_score
#%%importar datos
data_file='BaseDatosMicroestructuras-USDMXN.csv'
data=pd.read_csv(data_file,header=0)
data.columns=['Fecha','Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','IPC','Tasa_interes_referencia',
                      'Estabilidad_politica','USDMXN']
#%% descripción de datos
data_1=data.iloc[0:190,1:8]
quick_report=data.describe()
columns=pd.DataFrame(list(data.columns.values),columns=['Col names'])
data_type=pd.DataFrame(data.dtypes,columns=['Data type'])
quick_report_obj=data.describe(include=['object']).transpose()

#%% matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,0:7].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','IPC','Tasa_interes_referencia',
                      'Estabilidad_politica','USDMXN']
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','IPC','Tasa_interes_referencia',
                      'Estabilidad_politica','USDMXN'] 

#%% ajustes DATOS
data.Estabilidad_politica[204:209]=data.Estabilidad_politica[203]
data.Cuenta_corriente_milesdeUSD[204:209]=data.Cuenta_corriente_milesdeUSD[203]
data.Deuda_publica_neta_en_USD[204:209]=data.Deuda_publica_neta_en_USD[203]

#%%  cuarto Ajustes regresion 
#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(1,3,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
correlacion.index=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con petroleo, ipc,tasa,est.política

x=data_1.iloc[:,(1,3,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=2).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)

#### prueba regresion######

x=data.iloc[190:207,(2,4,5,6)]
xa=plf(degree=2).fit_transform(x)
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)

real=pd.DataFrame(data.USDMXN[190:207])
coefficient_of_dermination = r2_score(real,yg)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
resul=yg.append(real)