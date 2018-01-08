# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:46:05 2017

@author: if692068
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt

#%%
accidente_file= 'Accidents_2015.csv'
accidentes=pd.read_csv(accidente_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
                       
 #%%
minidata=accidentes.iloc[0:10,[2,3,6,7,5,9]]

#%%
minidata_dummy=minidata.iloc[:,0:4]
temp=pd.get_dummies(minidata['Accident_Severity'])   
temp.columns=['Severity_2','Severity_3']

minidata_dummy=minidata_dummy.join(temp)
temp=pd.get_dummies(minidata['Day_of_Week'])   
temp.columns=['Sunday','Monday','Tuesday','Thursday','Friday']
minidata_dummy=minidata_dummy.join(temp)
                    
#%% normalizar datos

                    