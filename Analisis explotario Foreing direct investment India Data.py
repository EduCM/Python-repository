# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:58:00 2017

@author: if692068
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

#%%
india_file= 'FDI_in_India.csv'
india=pd.read_csv(india_file,
                       header=0,
                       sep=',',
                       index_col=0,
                       parse_dates=False,
                       skip_blank_lines=True)
                       
india.head()

#%%data quality report
india=india.transpose()
columns=list(india.columns.values)
data_type=pd.DataFrame(india.dtypes,columns=['Data type'])
missing_data_counts=pd.DataFrame(india.isnull().sum(),columns=['missing values'])                     
present_data_couts=pd.DataFrame(india.count(),columns=['present values'])
unique_values_counts=pd.DataFrame(columns=['unique values'])

for v in list(india.columns.values):
    unique_values_counts.loc[v]=[india[v].nunique()]         


minimum_values=pd.DataFrame(columns=['minimum values'])         
for v in list(india.columns.values):
   try: 
    minimum_values.loc[v]=[india[v].min()]    
   except:
    pass                    

max_values=pd.DataFrame(columns=['maximun values'])         
for v in list(india.columns.values):
    max_values.loc[v]=[india[v].max()]                           
    
#%%
data_quality_report=data_type.join(missing_data_counts).join(present_data_couts).join(unique_values_counts).join(minimum_values).join(max_values)    

#%% quick report
descrip=india.describe()
quick_report=india.describe().transpose()


#%%gráfica pastel inversion por sector
suma=quick_report['mean'].sum()
propor=quick_report['mean']/suma

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = columns
fracs = propor

pie(fracs,labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

title('Inversion por sector', bbox={'facecolor':'0.8', 'pad':5})

show()

#%%crecimiento promedio anual
india=india.replace(0,0.01)     
crecimiento=india.iloc[16,:]/india.iloc[0,:]
crecimiento=pd.DataFrame(crecimiento).transpose()
cpa=crecimiento**(1/17.0)-1
cpa=cpa.transpose()

#%% inversion anual total
Invt=pd.DataFrame(india.sum(axis=1))
años= list([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
plt.plot(años,Invt.loc[:,0],'green')
plt.ylabel('Monto Anual Total(MDD)')
plt.show()
df=Invt.pct_change(periods=1)
crecimientotal=df.drop(df.index[0])
años= list([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016])
cero=pd.DataFrame(np.full((1,16),0)).transpose()
plt.plot(años,crecimientotal.loc[:,0],'red',años,cero,'b--')
#plt.plot(años,pd)
plt.ylabel('Variación anual')
plt.show()

#%% correlación 
corr=plt.matshow(india.corr())

corr = india.corr()
fig,ax = plt.subplots(figsize=(63, 63))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);

