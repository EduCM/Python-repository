# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:30:26 2017

@author: Edu
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
#%%
file= 'tpeunacional2015.csv'

data=pd.read_csv(file,
                       header=0,
                       sep=',',
                       parse_dates=False,
                       skip_blank_lines=True,
                       encoding='latin-1')
                       
                       
data.head()
#%%
columns=pd.DataFrame(list(data.columns.values),columns=['Col names'])
#%%
data_type=pd.DataFrame(data.dtypes,columns=['Data type'])
                     
#%%
missing_data_counts=pd.DataFrame(data.isnull().sum(),columns=['missing values'])                     
                      
#%%
present_data_couts=pd.DataFrame(data.count(),columns=['present values'])

#%%
unique_values_counts=pd.DataFrame(columns=['unique values'])
for v in list(data.columns.values):
    unique_values_counts.loc[v]=[data[v].nunique()]         

#%%

minimum_values=pd.DataFrame(columns=['minimum values'])         
for v in list(data.columns.values):
   try: 
    minimum_values.loc[v]=[data[v].min()]    
   except:
    pass                    
#%%

max_values=pd.DataFrame(columns=['maximun values'])         
for v in list(data.columns.values):
    max_values.loc[v]=[data[v].max()]                           
    
#%%
data_quality_report=data_type.join(missing_data_counts).join(present_data_couts).join(unique_values_counts).join(minimum_values).join(max_values)    

#%%
quick_report=data.describe(include=['object']).transpose()
                     
 #%%DUMMYES
data2=data.iloc[:,1:8]
temp=pd.get_dummies(data['Mes'])   
data2=data2.join(temp)

#%% agrupamiento 1 
trabajadores1=data.groupby(['Mes','Entidad_Federativa','Division_de_Actividad']).agg({'Trabajadores_eventuales_urbanos':np.sum})
trabajadores2=data.groupby(['Mes','Entidad_Federativa','Division_de_Actividad']).agg({'Trabajadores_permanentes':np.sum})
trabajadores3=data.groupby(['Mes','Entidad_Federativa','Division_de_Actividad']).agg({'Trabajadores_eventuales_del_campo':np.sum})

#%%agrupamiento 1
trabajadores=trabajadores1.merge(trabajadores2,left_index=True,right_index=True)
trabajadores=trabajadores.merge(trabajadores3,left_index=True,right_index=True)

#%%agrumpamiento 1
trabajadores['encabezado'] = trabajadores.index

trabajadores = trabajadores.set_index([np.arange(0,1536,1)])

a=trabajadores.iloc[:,0:3]
a_norms=(a-a.mean(axis=0))/a.std(axis=0)   
b=pd.DataFrame({'enero':np.zeros(1536),
                  'febrero':np.zeros(1536),
                  'marzo':np.zeros(1536),
                  'abril':np.zeros(1536),
                  'mayo':np.zeros(1536),
                  'junio':np.zeros(1536)})
                  
b.abril[0:256]=1 
b.enero[256:512]=1 
b.febrero[512:768]=1 
b.junio[768:1024]=1 
b.marzo[1024:1280]=1 
b.mayo[1280:1536]=1                 
 
trabajadoresfinal=a_norms.merge(b,left_index=True,right_index=True)

matrizsumilitud=sc.squareform(sc.pdist(trabajadoresfinal,'euclidean'))

#%% clustering 

k=hierarchy.linkage(trabajadoresfinal,'complete')
plt.figure(figsize=(25,10))
plt.title('dendograma completo')
plt.ylabel('distancia')
plt.xlabel('indice de la muestra')
dk=hierarchy.dendrogram(k)
plt.show()

#%% gráfica de 'codo'
last=k[-10:,2]
last_rev=last[::-1]
idxs=np.arange(1,len(last_rev)+1)
plt.plot(idxs,last_rev)

aceleracion=np.diff(last)
aceleracion_rev=aceleracion[::-1]
plt.plot(idxs[:-1]+1,aceleracion_rev)
plt.show()

#%% como saber que datos pertenece a su respectivo cluster 
j=6 # clusters
index=hierarchy.fcluster(k,j,criterion='maxclust')

# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#para que veas la diff de random y kmeans ++

#%% algoritmo kmeans

inercia = np.zeros(15)
for k in np.arange(1,15):
    model = KMeans(n_clusters = k,
               init = "random",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)

    model = model.fit(trabajadoresfinal)
    inercia[k] = model.inertia_

plt.plot(np.arange(1,15), inercia[1:])

aceleration = np.diff(inercia)
plt.plot(np.arange(2,15),-1*aceleration[1:])
plt.show()

#%%algoritmo kmeans
inercia = np.zeros(15)
for k in np.arange(1,15):
    model = KMeans(n_clusters = k,
               init = "k-means++",
               max_iter = 300,
               n_init = 10,
               n_jobs = 1)

    model = model.fit(trabajadoresfinal)
    inercia[k] = model.inertia_

plt.plot(np.arange(1,15), inercia[1:])

aceleration = np.diff(inercia)
plt.plot(np.arange(2,15),-1*aceleration[1:])
plt.show()

#%% info estadist. extra

asegurados_por_estado=data.groupby(['Entidad_Federativa']).agg({'Trabajadores_Asegurados':np.sum})
descrip_asegurados=asegurados_por_estado['Trabajadores_Asegurados'].describe()

import operator
min_index_estado, min_value_estado = min(enumerate(asegurados_por_estado.Trabajadores_Asegurados), key=operator.itemgetter(1))
max_index_estado, max_value_estado = max(enumerate(asegurados_por_estado.Trabajadores_Asegurados), key=operator.itemgetter(1))


asegurados_por_actividad=data.groupby(['Division_de_Actividad']).agg({'Trabajadores_Asegurados':np.sum})
descrip_asegurados_act=asegurados_por_actividad['Trabajadores_Asegurados'].describe()

import operator
min_index_act, min_value_act = min(enumerate(asegurados_por_actividad.Trabajadores_Asegurados), key=operator.itemgetter(1))
max_index_act, max_value_act = max(enumerate(asegurados_por_actividad.Trabajadores_Asegurados), key=operator.itemgetter(1))