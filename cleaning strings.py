# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 09:34:40 2017

@author: if692068
"""

import pandas as pd
from titlecase import titlecase
import string

#%%
lc=pd.DataFrame({'people':["alfonso a,guilar","edu6ardo castillo","jocelyn-medel","VICTOR ramos _ _ _"],
                  'edad':[24,35,46,57],
                  'ssn':['6439','689 24 9939','306-05-2792','99922a45832'],
                  'birth_date':['2/12/54','05/07/1958','01-26-1956','19xx-10-23'],
                  'marital':['soltero','casado','viudo','divorciado']})
#%%

exclude=string.punctuation
def remove_punctuation(x):
    try:
       x=''.join(ch for ch in x if ch not in exclude)
    except:
       pass
    return x
        
#%%
def remove_digits(x):
    try:
       x=''.join(ch for ch in x if ch not in string.digits)
    except:
       pass
    return x        
    
#%%
def remove_whitespaces(x):
    try:
       x=''.join(x.split())
    except:
       pass
    return x    

#%%
def replace_text(x,to_replace,replacement):
    try:
       x=x.replace(to_replace,replacement)
    except:
       pass
    return x  
#%%
def uppercas_text(x):
    try:
         x=x.upper()
    except:
       pass
    return x   

#%%
def lowercas_text(x):
    try:
         x=x.lower()
    except:
       pass
    return x   
#%%
 def titlecase_text(x):
    try:
         x=titlecase(x)
    except:
       pass
    return x      
#%%
lc.people[2]=replace_text(lc.people[2],'-',' ')    
lc.people=lc.people.apply(replace_text,args=('-',' '))
lc.people=lc.people.apply(remove_punctuation)          
lc.people=lc.people.apply(remove_digits)      
lc.people=lc.people.apply(remove_whitespaces)  
lc.people=lc.people.apply(uppercas_text)  
lc.people=lc.people.apply(lowercas_text)  
lc.people=lc.people.apply(titlecase_text)  