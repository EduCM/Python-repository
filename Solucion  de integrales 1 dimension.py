# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:21:38 2016

@author: Edu
"""
"trapecios"
import numpy

def trapecios(f,limiteinf,limitesup,numtrapecios):
    x=numpy.arange(limiteinf,limitesup,((limitesup-limiteinf)/numtrapecios))
    integral=((limitesup-limiteinf)/numtrapecios)*((f(limiteinf)+f(limitesup)/2)+sum(f(x)))
    return integral 
   
t=trapecios(lambda x:x**2,3,5,100000)

"montecarlo"
def montecarlo(f,limiteinf,limitesup,n):
    integral=(limitesup-limiteinf)*numpy.mean(f(numpy.random.uniform(limiteinf,limitesup,n)))
    return integral 

m=montecarlo(lambda x:x**2,3,5,100000)

"simpson 1/3"
def simpson(f,limiteinf,limitesup):
    integral=((limitesup-limiteinf)/6)*(f(limiteinf)+4*f((limiteinf+limitesup)/2)+f(limitesup))
    return integral 

s=simpson(lambda x:x**2,3,5)

