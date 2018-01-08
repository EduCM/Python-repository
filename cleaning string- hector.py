# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:29:55 2017

@author: Hector
"""

import pandas as pd
import titlecase
import string
#%% limpiar base de datos cadenas de texto
lc = pd.DataFrame({
                   "people":["Alfonso A,guilar","edu6ardo Castillo", "jocelyn-medel","VICTOR ramos _ _ _"],
                   "age":[24,35,46,57],
                   "ssn":["6439","689 24 9939", "306-05-2792","99922a45832"],
                   "birth_date":["2/12/54","05/07/1958","01-26-1956","19xx-10-23"],
                   "marital status":["soltero","casado","viudo","divorciado"]})
#%%
exclude = string.punctuation

#%%
def remove_punctuation(x):
    try:
        x = "".join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x
#%%
remove_punctuation("manuel -lopez*")
##%%
#lc.people
#lc.people = lc.people.apply(remove_punctuation)
#lc
#%%
def remove_digits(x):
    try:
        x = "".join(ch for ch in x if ch not in string.digits)
    except:
        pass
    return x
#%%
#lc.people
#lc.people = lc.people.apply(remove_digits)
#lc
#%%
#lc.people[0]
#lc.people[0].split()
#"".join(lc.people[0].split())
#%%
def remove_white_spaces(x):
    try:
        x = "".join(x.split)
    except:
        pass
    return x
#%%
def replace_text(x,to_replace,replacement):
    try:
        x = x.replace(to_replace,replacement)
    except:
        pass
    return x
#%%
def uppercase_text(x):
    try:
        x = x.upper()
    except:
        pass
    return(x) 
#%%
def lowercase_text(x):
    try:
        x = x.lower()
    except:
        pass
    return(x) 
#%%
lc.people
lc.people[2] = replace_text(lc.people[2],"-"," ")
lc.people = lc.people.apply(remove_punctuation)
lc.people = lc.people.apply(remove_digits)
lc.people = lc.people.apply(uppercase_text)
lc.people = lc.people.apply(lowercase_text)
lc.people
 




