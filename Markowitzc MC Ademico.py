# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:33:38 2016

@author: Edu
"""
miuA=.3
miuB=.1
sigA=.23
sigB=.17
rho=-.5
n=1000
import numpy
import matplotlib.pyplot as plt

def nube(miuA,miuB,sigA,sigB,Rho,n):
    rendimientos=numpy.matrix([miuA,miuB])
    fila1=[sigA**2,sigB*sigA*rho]
    fila2=[sigB*sigA*rho,sigB**2]
    matrizcov = numpy.matrix([fila1,fila2])
    activos=len(matrizcov)
    randnum=numpy.random.uniform(0,1,size=(activos,n))
    randfin= randnum.sum(axis=0)
    randnum= randnum/randfin
    vEsperanza=rendimientos@randnum
    mcovpart=randnum.transpose()@matrizcov@randnum
    vDesvSinfor=numpy.sqrt(numpy.diag(mcovpart))
    vDesvSinfor=numpy.matrix(vDesvSinfor)
    return(vDesvSinfor,vEsperanza,randnum)
x,y,z=nube(miuA,miuB,sigA,sigB,rho,n)

i = x.argmin(1)
peso1,peso2=z[:,i]
ll = plt.plot(x,y,'ro',x[0,i],y[0,i], 'bs')
plt.axis([.05, .26, .05, .35])
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlabel('Desviación')
plt.ylabel('Rendimiento')
plt.title('Frontera Eficiciente')
plt.grid(True)
plt.show()
    

    