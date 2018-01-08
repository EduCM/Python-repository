# -*- coding: utf-8 -*-
"""
Mi primer lectura de archivo csv
"""

import numpy as np
import pandas as pd

#%%
accidente_file= 'Casualties_2015.csv'
accidentes=pd.read_csv(accidente_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
                       
accidentes.head()

#%%

columns=pd.DataFrame(list(accidentes.columns.values),columns=['Col names'])

#%%
data_type=pd.DataFrame(accidentes.dtypes,columns=['Data type'])
                     
#%%
missing_data_counts=pd.DataFrame(accidentes.isnull().sum(),columns=['missing values'])                     
                      
#%%
present_data_couts=pd.DataFrame(accidentes.count(),columns=['present values'])

#%%
unique_values_counts=pd.DataFrame(columns=['unique values'])
for v in list(accidentes.columns.values):
    unique_values_counts.loc[v]=[accidentes[v].nunique()]         

#%%

minimum_values=pd.DataFrame(columns=['minimum values'])         
for v in list(accidentes.columns.values):
   try: 
    minimum_values.loc[v]=[accidentes[v].min()]    
   except:
    pass                    
#%%

max_values=pd.DataFrame(columns=['maximun values'])         
for v in list(accidentes.columns.values):
    max_values.loc[v]=[accidentes[v].max()]                           
    
#%%
data_quality_report=data_type.join(missing_data_counts).join(present_data_couts).join(unique_values_counts).join(minimum_values).join(max_values)    

print "\ Data Quality report"
print "Total records:()".format(len(accidentes.index))
data_quality_report