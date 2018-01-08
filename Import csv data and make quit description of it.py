# -*- coding: utf-8 -*-
"""
Mi primer lectura de archivo csv
"""

import numpy as np
import pandas as pd

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

