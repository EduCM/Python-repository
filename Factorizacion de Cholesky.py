# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:23:13 2016

@author: PC
"""

import numpy
matriz = numpy.matrix('1 .8 .3;.5 1 .6;.6 .5 1')
def chole(matriz,n):
    chol = numpy.linalg.cholesky
    c = numpy.matrix(chol(matriz))
    x = numpy.random.normal(0, 1, size=(n,c.shape[0]))
    d= c@x.transpose()
    return d.transpose()
    
aa=chole(matriz,1000)
revi = numpy.cov(aa.transpose())
