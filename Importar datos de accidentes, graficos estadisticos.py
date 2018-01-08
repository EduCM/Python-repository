
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

accidents_file='Accidents_2015.csv'
accidents=pd.read_csv(accidents_file,
                      header=0,
                      sep=',',
                      index_col=0,
                      parse_dates=False,
                      skip_blank_lines=True)
 
#%%
count_vehicle=accidents.groupby('Date').agg({'Number_of_Vehicles':np.sum})
count_casualties=accidents.groupby('Date').agg({'Number_of_Casualties':np.sum})

#%%


data_plot= [count_casualties['Number_of_Casualties'],count_vehicle['Number_of_Vehicles']]                

#%%
fig=plt.figure(1,figsize=(9,6))
ax=fig.add_subplot(111)
# los primeros dos numeros del subplot son el tamaño de la "matriz de graficas" y el 3ro es el numero de subgráfica
bp=ax.boxplot(data_plot)

for median in bp['medians']:
    median.set(color= '#b2df8a',linewidth=10)
    
for cap in bp['caps']:
    cap.set(color='#7570b3',linewidth=2)

for fli in bp['fliers']:
    fli.set(marker='o',color='#7570b3',alpha=0.5)

ax.set_xticklabels(['Casualties','Vehicles'])
fig.savefig('figura1.png')    
   
plt.show()    

#%%

accidents.boxplot(column='Light_Conditions')
accidents.boxplot(column='Light_Conditions',by='Weather_Conditions')
accidents.boxplot(column='Number_of_Vehicles',by='Day_of_Week')

#%%
casualty_count=accidents.groupby('Day_of_Week').Number_of_Casualties.count()
## esta forma de agrupar es para funciones preexistentes de numpy o panda, number of csualties es la columna que agrupara por dias de la semana
casualty_prob=casualty_count/accidents.groupby('Day_of_Week').Number_of_Casualties.sum()
fig=plt.figure(1,figsize=(9,6))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Day of Week')
ax1.set_ylabel('Casualty count')
ax1.set_title('casualties by day of the week')
casualty_count.plot(kind='bar')

ax1=fig.add_subplot(122)
ax1.set_xlabel('Day of Week')
ax1.set_ylabel('Casualty prob')
ax1.set_title('casualties prob by day of the week')
casualty_prob.plot(kind='bar')
