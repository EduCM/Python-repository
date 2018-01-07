# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 14:09:43 2016

@author: PC
"""
miuA=.3
miuB=.2
sigA=.7
sigB=.8
w1=.5
w2=.5
rho=0
stA=5
stB=7
n=10000
Te=30
t=0

"probabilidad caida precio portafolio academico"
import numpy
def portafolioac(miuA,miuB,sigA,sigB,w1,w2,rho,stA,stB,n,Te,t):
    fila1=[sigA**2,sigB*sigA*rho]
    fila2=[sigB*sigA*rho,sigB**2]
    matriz = numpy.matrix([fila1,fila2])
    chol = numpy.linalg.cholesky
    c = numpy.matrix(chol(matriz))
    z = numpy.random.normal(0, 1, size=(n,2))
    x= c@z.transpose()
    x=x.transpose()
    STA=stA*numpy.exp((miuA-(sigA**2)/2)*(Te-t)+(sigA*numpy.sqrt(Te-t)*(x[:,0])))
    STB=stB*numpy.exp((miuB-(sigB**2)/2)*(Te-t)+(sigB*numpy.sqrt(Te-t)*(x[:,1])))
    po=w1*stA + w2*stB
    pT=w1*STA + w2*STB
    rend=(pT/po)
    rend=rend[rend<=1]
    return(rend.shape[1]/n)
    
prob=portafolioac(miuA,miuB,sigA,sigB,w1,w2,rho,stA,stB,n,Te,t)