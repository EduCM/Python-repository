# -*- coding: utf-8 -*-
"""
Mi primer lectura de archivo csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
accidente_file= 'Accidents_2015.csv'
accidentes=pd.read_csv(accidente_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
                       
accidentes.head()

#%%
descrip=accidentes.describe()

descrip_speed=accidentes['Speed_limit'].describe()

quick_report1=accidentes.describe().transpose()

#%%

quick_report2=accidentes.describe(include=['object']).transpose()
## incluye variables tipo objeto

#%%
quick_report3=accidentes.mode().transpose()

## para la moda 

#%%
accidentes_por_dia=pd.value_counts(accidentes['Date'])

#%%
##estadisticas de numero de vehiculos

print "mean value:{}".format(accidentes['Number_of_Vehicles'].mean())
print "min value:{}".format(accidentes['Number_of_Vehicles'].min())
print "max value:{}".format(accidentes['Number_of_Vehicles'].max())
print "mode value:{}".format(accidentes['Number_of_Vehicles'].mode())
print "std value:{}".format(accidentes['Number_of_Vehicles'].std())

#%%

vehicle_counts=accidentes.groupby('Date').agg({'Number_of_Vehicles':np.sum})
casualty_counts=accidentes.groupby('Date').agg({'Number_of_Casualties':np.sum})
#%%
##para pegar las dos matrices
vehicle_casualty=casualty_counts.merge(vehicle_counts,left_index=True,right_index=True)
accidentes_por_dia=pd.DataFrame(accidentes_por_dia)
vehicle_casualty=vehicle_casualty.merge(accidentes_por_dia,left_index=True,right_index=True)
##agg es como un apply

#%%Gráficos

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30)
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('frequency')
plt.show()

#%%Gráficos

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30,normed=True)
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('Probabilidad')
plt.show()

#%%Gráficos

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30,normed=True,cumulative=True)
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('Probabilidad')
plt.show()

#%%Gráficos

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30,normed=True,histtype='step')
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('Probabilidad')
plt.show()

#%%Gráficos

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30,normed=True,histtype='step')
plt.hist(casualty_counts['Number_of_Casualties'],bins=30,normed=True,histtype='step')
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('Probabilidad')
plt.show()

#%%

plt.hist(vehicle_counts['Number_of_Vehicles'],bins=30,normed=True,histtype='stepfilled',label='vehicles',color='b',alpha=.5)
plt.hist(casualty_counts['Number_of_Casualties'],bins=30,normed=True,histtype='stepfilled',label='casualties',color='r',alpha=.5)
plt.legend()
plt.title('histograma de numero de vehiculos')
plt.xlabel('numero de vehiculos')
plt.ylabel('Probabilidad')
plt.show()











