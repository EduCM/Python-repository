# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 06:33:42 2017

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
         año, evo_porc.iloc[:,5], 'r',
        año, evo_porc.iloc[:,0], 'g',
        año, evo_porc.iloc[:,1], '')

plt.title("Evolucion % anual")
plt.xlabel("Año")
plt.ylabel("Variación")
plt.legend(['Casos de VIH/SIDA','Egresos hospitalarios','Esperanza de vida al nacer',
            'Muertes maternas', 'Casos de Influenza A H1N1', 'Casos de denge'], 
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
plt.legend([],bbox_to_anchor=(1, 1), loc=2)           
plt.show()