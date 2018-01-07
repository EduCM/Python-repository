# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:28:29 2016

@author: Edu
"""

import numpy
import math
from pandas.io.data import DataReader
from datetime import datetime
from scipy.stats import norm

alfa = DataReader('ALFAA.MX',  'yahoo', datetime(2015,4,9), datetime(2016,4,9))
alfa=alfa['Adj Close']
pinfra = DataReader('PINFRA.MX',  'yahoo', datetime(2015,4,9), datetime(2016,4,9))
pinfra=pinfra['Adj Close']
ipc = DataReader('^MXX',  'yahoo', datetime(2015,4,9), datetime(2016,4,9))
ipc=ipc['Adj Close']

logalfa=numpy.log(alfa.astype('float64')/alfa.astype('float64').shift(1))
logpinfra=numpy.log(pinfra.astype('float64')/pinfra.astype('float64').shift(1))
logipc=numpy.log(ipc.astype('float64')/ipc.astype('float64').shift(1))

ralfa=numpy.mean(logalfa[1:len(logalfa)])
rpinfra=numpy.mean(logpinfra[1:len(logpinfra)])
ripc=numpy.mean(logipc[1:len(logipc)])

valfa=numpy.std(logalfa[1:len(logalfa)])
vpinfra=numpy.std(logpinfra[1:len(logpinfra)])
vipc=numpy.std(logipc[1:len(logipc)])

"parametros entrada"

r=ripc                    #Rendimiento esperado el activo
sigma=vipc                #Volatilidad del rendimiento del activo
k=45500                     #Precio Strike
St=ipc[len(ipc)-1]       #Precio Spot
Te=50                    #Tiempo Final
t=0                       #Tiempo inicial
n=1000000             #Numero de simulaciones para Montecarlo y Analítico
tipo=1               #1 para Call y 2 para Put
alfae=1                #Parámetro de elasticidad de varianza para Euler
dt=.01                #Tañmaño de pasos para Euler
sim=2000              #numero de simulaciones para Euler

"solucion black-scholes"

def bs(r,sigma,k,St,Te,t,tipo):
    if tipo==1:
        CT=St*norm.cdf((numpy.log(St/k)+(r+sigma**2/2)*(Te-t))/(sigma*(Te-t)**(.5)))-k*numpy.exp(-r*(Te-t))*norm.cdf((numpy.log(St/k)+(r-sigma**2/2)*(Te-t))/(sigma*(Te-t)**(.5)))
    else:
        CT=-St*norm.cdf(-(numpy.log(St/k)+(r+sigma**2/2)*(Te-t))/(sigma*(Te-t)**(.5)))+k*numpy.exp(-r*(Te-t))*norm.cdf(-(numpy.log(St/k)+(r-sigma**2/2)*(Te-t))/(sigma*(Te-t)**(.5)))
        
    return CT    
bs=bs(r,sigma,k,St,Te,t,tipo)

"solucion montecarlo"

def mc(r,sigma,k,St,Te,t,n,tipo):
    if tipo==1:
        a=numpy.log(k/St)
        b=(r-sigma**2/2)*(Te-t)+6*sigma*numpy.sqrt(Te-t)
        h=(b-a)/n
        x=numpy.random.uniform(a,b,n)
    else:
        b=numpy.log(k/St)
        a=(r-sigma**2/2)*(Te-t)-6*sigma*numpy.sqrt(Te-t)
        h=(b-a)/n
        x=numpy.random.uniform(a,b,n)
        
    if tipo==1:
        Ct=(St*numpy.exp(x)-k)*1/(sigma*numpy.sqrt((Te-t)*2*math.pi))*numpy.exp(-(x-(r-sigma**2/2)*(Te-t))**2/(2*(sigma**2)*(Te-t)))
        Ct[Ct<0]= 0
    else:
        Ct=(k-St*numpy.exp(x))*1/(sigma*numpy.sqrt((Te-t)*2*math.pi))*numpy.exp(-(x-(r-sigma**2/2)*(Te-t))**2/(2*(sigma**2)*(Te-t)))
        Ct[Ct<0]=0
    Ct=h*numpy.exp(-r*(Te-t))*sum(Ct)
    return Ct

mc=mc(r,sigma,k,St,Te,t,n,tipo)

"montecarlo sobre solucion analítica"

def analit(r,sigma,k,St,Te,t,n,tipo):
    Zt=numpy.random.normal(0,1,n)
    if tipo==1:
        ST=St*numpy.exp((r-sigma**2/2)*(Te-t)+sigma*numpy.sqrt(Te-t)*Zt)-k #solo se simula el última valor en tiempo T( en lugar de toda la trayectoria)
        ST[ST<0]= 0 #cuando el precio simulado menos el precio strike es menor a 0, el valor es 0
        
    else:
        ST=k-St*numpy.exp((r-sigma**2/2)*(Te-t)+sigma*numpy.sqrt(Te-t)*Zt) #solo se simula el última valor en tiempo T( en lugar de toda la trayectoria)
        ST[ST<0]= 0 #cuando el precio simulado menos el precio strike es menor a 0, el valor es 0
    Ct=numpy.exp(-r*(Te-t))*numpy.mean(ST)
    return Ct

analit=analit(r,sigma,k,St,Te,t,n,tipo)

"Elasticidad constante de la varianza (Método de Euler)"

def euler(r,sigma,k,St,Te,alfa,dt,sim,tipo):
    t=numpy.arange(0,Te,dt)
    n= len(t)
    Xt=numpy.random.normal(0,1,size=(sim,n))
    Zt=numpy.random.normal(0,1,size=(sim,n))
    Xt[:,0]=St
    
    for i in range(1,n):
        Xt[:,i]= Xt[:,i-1] + dt*r*Xt[:,i-1]+sigma*((Xt[:,i-1])**alfae)*numpy.sqrt(dt)*Zt[:,i-1]

    if tipo==1:
        intm=Xt[:,n-1]
        intm=intm-k
        intm[intm<0]=0
    else:
        intm=Xt[:,n-1]
        intm=k-intm
        intm[intm<0]=0
    Ct=numpy.exp(-r*(Te))*numpy.mean(intm)
    return Ct

euler=euler(r,sigma,k,St,Te,alfa,dt,sim,tipo) 


    
