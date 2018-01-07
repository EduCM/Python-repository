import numpy
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt

alfa = DataReader('AMXL.MX',  'yahoo', datetime(2015,4,9), datetime(2016,4,9))
alfa=alfa['Adj Close']
logalfa=numpy.log(alfa.astype('float64')/alfa.astype('float64').shift(1))
logalfa=logalfa[1:len(logalfa)]
ralfa=numpy.mean(logalfa)
valfa=numpy.std(logalfa)

"parametros entrada"

mu=ralfa                    #Rendimiento esperado el activo
sigma=valfa                #Volatilidad del rendimiento del activo
St=alfa[len(alfa)-1]       #Precio Spot
Te=50                     #Tiempo Final
tipo = 1
strike=14.5
n=1000

def intesto(mu,sigma,Te,n,tipo,st):
    dt = .01
    k = round(Te/dt)
    dB = numpy.sqrt(dt) * numpy.random.normal(0,1,(n,k))
    Res = numpy.random.rand(n,k+1)
    Res[:,0] = st
   
    if tipo==1:
        for i in range(1,k+1):
            Res[:,i] = Res[:,i - 1] + (mu * Res[:,i - 1] * dt) + (sigma * Res[:,i - 1] * dB[:,i-1])
    
    if tipo==2:
        for i in range(1,k+1):
            Res[:,i] = Res[:,i - 1] + mu * Res[:,i - 1] * dt + sigma * Res[:,i - 1] * dB[:,i-1] + 0.5 * sigma**2 * Res[:,i - 1] * (dB[:,i-1]**2 - dt)

    return(Res)
    
Respuesta = intesto(mu,sigma,Te,n,tipo,St) #Tipo 1 es para Euler-Maruyama y 2 para Milstein
#ll = plt.plot(Respuesta.transpose())
#plt.show()

Call = Respuesta[:,Respuesta.shape[1]-1]-strike
Call[Call<0]=0
Call=numpy.exp(-mu*Te)*numpy.mean(Call)

Put = strike - Respuesta[:,Respuesta.shape[1]-1]
Put[Put<0]=0
Put=numpy.exp(-mu*Te)*numpy.mean(Put)