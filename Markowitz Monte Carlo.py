# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:11:58 2016

@author: PC
"""
import numpy
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt

alfa = DataReader('ALFAA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
alfa=alfa['Adj Close']
pinfra = DataReader('PINFRA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
pinfra=pinfra['Adj Close']
alpeka = DataReader('ALPEKA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
alpeka=alpeka['Adj Close']
sanmexb = DataReader('SANMEXB.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
sanmexb=sanmexb['Adj Close']
gfinburo = DataReader('GFINBURO.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
gfinburo=gfinburo['Adj Close']

ralfa=numpy.log(alfa.astype('float64')/alfa.astype('float64').shift(1))
rpinfra=numpy.log(pinfra.astype('float64')/pinfra.astype('float64').shift(1))
ralpeka=numpy.log(alpeka.astype('float64')/alpeka.astype('float64').shift(1))
rsanmexb=numpy.log(sanmexb.astype('float64')/sanmexb.astype('float64').shift(1))
rgfinburo=numpy.log(gfinburo.astype('float64')/gfinburo.astype('float64').shift(1))

rendimientos=numpy.matrix([ralfa[1:len(ralfa)],rpinfra[1:len(rpinfra)],ralpeka[1:len(ralpeka)],rsanmexb[1:len(rsanmexb)],rgfinburo[1:len(rgfinburo)]])

def nube(rendimientos,num):
    matrizcov=numpy.cov(rendimientos)
    activos=rendimientos.shape[0]
    randnum=numpy.random.uniform(0,1,size=(activos,num))
    randfin= randnum.sum(axis=0)
    randnum= randnum/randfin
    rmean=rendimientos.mean(axis=1)
    vEsperanza=rmean.transpose()@randnum
    vEsperanza=(numpy.array(vEsperanza)).transpose()
    mcovpart=randnum.transpose()@matrizcov@randnum
    vDesvSinfor=numpy.sqrt(numpy.diag(mcovpart))
    return(vDesvSinfor,vEsperanza,randnum)
x,y,z=nube(rendimientos,10000)

i = x.argmin(0)
peso1,peso2,peso3,peso4,peso5=z[:,i]
ll = plt.plot(x,y,'ro',x[i],y[i,0], 'bs')
plt.axis([0.045, .2, -.105, .1])
plt.text(.115, 0, r'$\mu=0.00768,\ \sigma=0.0786$')
plt.xlabel('Desviación')
plt.ylabel('Rendimiento')
plt.title('Frontera Eficiciente')
plt.grid(True)
plt.show()


