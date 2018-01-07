# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:51:23 2016

@author: Edu
"""
import numpy

"trapecios"
def trapecios2(f,limiteinf1,limitesup1,limiteinf2,limitesup2,numtrapecios):
    x=numpy.arange(limiteinf1,limitesup1,((limitesup1-limiteinf1)/numtrapecios))
    y=numpy.arange(limiteinf2,limitesup2,((limitesup2-limiteinf2)/numtrapecios))
    xv,yv = numpy.meshgrid(x, y)
    y=f(xv,yv)
    integral=numpy.mean(numpy.mean(y))*(limitesup1-limiteinf1)*(limitesup2-limiteinf2)
    return integral
    
t2=trapecios2(lambda x,y:x**2+y**2,1,2,2,3,2000)

"montecarlo"

def montecarlo2(f,limiteinf1,limitesup1,limiteinf2,limitesup2,n):
    x=numpy.random.uniform(limiteinf1,limitesup1,n)
    y=numpy.random.uniform(limiteinf2,limitesup2,n)
    xv,yv = numpy.meshgrid(x, y)
    y=f(xv,yv)
    integral=numpy.mean(numpy.mean(y))*(limitesup1-limiteinf1)*(limitesup2-limiteinf2)
    return integral
    
m2=montecarlo2(lambda x,y:x**2+y**2,1,2,2,3,2000)