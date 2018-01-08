# -*- coding: utf-8 -*-
"""
Created on Sun May  7 14:51:10 2017

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
data_1=data.iloc[0:204,1:8]
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
suma=np.absolute(correlacion).sum(axis=0)-np.absolute(correlacion).USDMXN
#%% Regresion multiple

x=data_1.iloc[:,0:6]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)

#### prueba regresion######
x=data.iloc[204:207,1:7]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)
#%% ajustes DATOS
data.Estabilidad_politica[204:209]=data.Estabilidad_politica[203]
data.Cuenta_corriente_milesdeUSD[204:209]=data.Cuenta_corriente_milesdeUSD[203]
data.Deuda_publica_neta_en_USD[204:209]=data.Deuda_publica_neta_en_USD[203]

#%%  primer Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(0,1,2,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia',
                      'Estabilidad_politica']
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia',
                      'Estabilidad_politica']  
suma=np.absolute(correlacion).sum(axis=0)

####### regresion sin "IPC"

x=data_1.iloc[:,(0,1,2,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)

#### prueba regresion######

x=data.iloc[204:207,(1,2,3,5,6)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  segundo Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(0,1,2,4)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia']
                      
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion sin "IPC" ni "Estabilidad política"

x=data_1.iloc[:,(0,1,2,4)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)

#### prueba regresion######

x=data.iloc[204:207,(1,2,3,5)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  tercer Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(1,2,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia','Estabilidad_politica']
                      
correlacion.index=['Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','Tasa_interes_referencia','Estabilidad_politica']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con petroleo, deuda, tasa y estabilidad política

x=data_1.iloc[:,(1,2,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)

#### prueba regresion######

x=data.iloc[204:207,(2,3,5,6)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

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

x=data.iloc[204:207,(2,4,5,6)]
xa=plf(degree=2).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)
### reduccion post-modelado, no mejoró nada

#w1=reglog.coef_
#plt.bar(np.arange(len(w1[0])),w1[0])
#wabs1=np.abs(w1)
#plt.bar(np.arange(len(wabs1[0])),wabs1[0])
#
#umbral=0.01
#index=wabs1>umbral
#xas1=xa[:,index[0]]
#
#reglog1_2=linear_model.LinearRegression()
#reglog1_2.fit(xas1,y)
#
#coef=reglog1_2.coef_
#yg=reglog1_2.predict(xas1)
#yg=pd.DataFrame(yg)
#coefficient_of_dermination = r2_score(y,yg)


#%%  5to Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(0,1,2,3,4)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','IPC','Tasa_interes_referencia']
                      
                      
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                      'Deuda_publica_neta_en_USD','IPC','Tasa_interes_referencia']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con todas menos estabilidad política

x=data_1.iloc[:,(0,1,2,3,4)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(1,2,3,4,5)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  sexto Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(0,1,3,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                     'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
                      
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                     'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con todas menos DEUDA

x=data_1.iloc[:,(0,1,3,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(1,2,4,5,6)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  septimo Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(0,1,3,4)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                     'IPC','Tasa_interes_referencia']
                      
                      
correlacion.index=['Cuenta_corriente_milesdeUSD','Mezcla_crudo_mexicano_USDxbarril',
                     'IPC','Tasa_interes_referencia']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con todas menos DEUDA y estabilidad política

x=data_1.iloc[:,(0,1,3,4)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(1,2,4,5)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  octavo Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(1,3,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
correlacion.index=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia','Estabilidad_politica']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con todas menos Cuenta corriente y deuda
x=data_1.iloc[:,(1,3,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(2,4,5,6)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  noveno Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(1,3,4)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia']
                      
correlacion.index=['Mezcla_crudo_mexicano_USDxbarril',
                      'IPC','Tasa_interes_referencia']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con petroleo,IPC y tasa
x=data_1.iloc[:,(1,3,4)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(2,4,5)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)

#%%  décimo Ajustes regresion 

#####   matriz correlacion
correlacion=np.corrcoef(data_1.iloc[:,(1,4,5)].transpose())
correlacion=pd.DataFrame(correlacion)
correlacion.columns=['Mezcla_crudo_mexicano_USDxbarril',
                      'Tasa_interes_referencia','Estabilidad_politica']
                      
correlacion.index=['Mezcla_crudo_mexicano_USDxbarril',
                      'Tasa_interes_referencia','Estabilidad_politica']
                      
suma=np.absolute(correlacion).sum(axis=0)

####### regresion con petroleo,tasa y estabilidad
x=data_1.iloc[:,(1,4,5)]
y=data_1.iloc[:,6:7] 

xa=plf(degree=1).fit_transform(x)
reglog=linear_model.LinearRegression()
reglog.fit(xa,y)# entrenar el modelo

coef=reglog.coef_
yg=reglog.predict(xa)
yg=pd.DataFrame(yg)
coefficient_of_dermination = r2_score(y,yg)
#### prueba regresion######

x=data.iloc[204:207,(2,5,6)]
xa=plf(degree=1).fit_transform(x)
yg=reglog.predict(xa)

yg=pd.DataFrame(yg)
yg.columns=['Prediccion']
real=pd.DataFrame(data.USDMXN[204:207])
resul=yg.append(real)