# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:58:00 2017

@author: if692068
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
creditos_file= 'creditcard.csv'
creditos=pd.read_csv(creditos_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
                       
creditos.head()

#%%data quality report

columns=pd.DataFrame(list(creditos.columns.values),columns=['Col names'])
data_type=pd.DataFrame(creditos.dtypes,columns=['Data type'])
missing_data_counts=pd.DataFrame(creditos.isnull().sum(),columns=['missing values'])                     
present_data_couts=pd.DataFrame(creditos.count(),columns=['present values'])
unique_values_counts=pd.DataFrame(columns=['unique values'])

for v in list(creditos.columns.values):
    unique_values_counts.loc[v]=[creditos[v].nunique()]         


minimum_values=pd.DataFrame(columns=['minimum values'])         
for v in list(creditos.columns.values):
   try: 
    minimum_values.loc[v]=[creditos[v].min()]    
   except:
    pass                    

max_values=pd.DataFrame(columns=['maximun values'])         
for v in list(creditos.columns.values):
    max_values.loc[v]=[creditos[v].max()]                           
    
#%%
data_quality_report=data_type.join(missing_data_counts).join(present_data_couts).join(unique_values_counts).join(minimum_values).join(max_values)    

#%% quick report
descrip=creditos.describe()
quick_report=creditos.describe().transpose()

#%%Gráficos

plt.hist(creditos['Amount'],bins=100,facecolor='green',range=[0, 2000])
plt.title('histograma del monto del prestamo')
plt.xlabel('Monto')
plt.ylabel('frequency')
plt.show()

plt.hist(creditos['Class'],bins=100)
plt.title('histograma de clases')
plt.xlabel('Clase')
plt.ylabel('frecuencia')
plt.show()

#%%conteo
conteo=float(len(creditos['Class']))
conteofraudes=creditos['Class'].sum()
conteovalidas=conteo-conteofraudes
porcfraudes=float(conteofraudes/conteo)*100

#%%

