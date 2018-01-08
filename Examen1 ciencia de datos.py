# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:20:06 2017

@author: Edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
enfermedades_file= 'enfermedades.csv'
enfermedades=pd.read_csv(enfermedades_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
#%% pregunta 4 sección A)
evo_porc=enfermedades.transpose()[::-1].pct_change()
evo_porc=evo_porc*100

año=np.array([10,11,12,13,14])

plt.plot(año, evo_porc.iloc[:,2], 'y',
         año, evo_porc.iloc[:,3], 'm',
         año, evo_porc.iloc[:,4], 'b',
         año, evo_porc.iloc[:,5], 'r')
plt.title("Evolucion % anual")
plt.xlabel("Año")
plt.ylabel("Variación")
plt.legend(['Casos de VIH/SIDA','Egresos hospitalarios','Esperanza de vida al nacer',
            'Muertes maternas'], 
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

plt.plot(año, evo_porc.iloc[:,1], 'm')             
plt.title("Evolucion % anual")
plt.xlabel("Año")
plt.ylabel("Variación")
plt.legend(['Casos de Influenza A H1N1'],bbox_to_anchor=(1, 1), loc=2)           
plt.show()

plt.plot(año, evo_porc.iloc[:,0], 'y')           
plt.title("Evolucion % anual")
plt.xlabel("Año")
plt.ylabel("Variación")
plt.legend(['Casos de denge'],bbox_to_anchor=(1, 1), loc=2)           
plt.show()

#%% pregunta 4 seccion B)
evo_porc_abs=np.abs(evo_porc)
suma=evo_porc_abs.sum(axis=0)
maximo=suma.argmax()

#%%pregunta 5 seccion C)
corre=evo_porc.corr()
medias=evo_porc.mean()

#%% análisis de componentes principales, pregunta 5

medias=enfermedades.mean(axis=0)
minidata=enfermedades-medias
data_cov=np.cov(minidata.transpose())
w,v=np.linalg.eig(data_cov)

index=np.argsort(w)[::-1]   
componentes=w[index[0:1]]
transform=v[:,index[0:1]]          
                 
data_new=np.matrix(minidata)*np.matrix(transform)
data_r=np.array(data_new)*np.matrix(transform.transpose())
medias=np.matrix(medias)
data_r=data_r+medias

resta_matrices=np.matrix(enfermedades)-data_r