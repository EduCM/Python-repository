
import numpy
mu = .1
sigma = .2
tipo = 2
strike=11
Te = 65/252
n=1000
def intesto(mu,sigma,Te,n,tipo):
    dt = 2**-9
    k = round(Te/dt)
    dB = numpy.sqrt(dt) * numpy.random.normal(0,1,(n,k))
    Res = numpy.random.rand(n,k+1)
    Res[:,0] = 10
   
    if tipo==1:
        for i in range(1,k+1):
            Res[:,i] = Res[:,i - 1] + (mu * Res[:,i - 1] * dt) + (sigma * Res[:,i - 1] * dB[:,i-1])
    
    if tipo==2:
        for i in range(1,k+1):
            Res[:,i] = Res[:,i - 1] + mu * Res[:,i - 1] * dt + sigma * Res[:,i - 1] * dB[:,i-1] + 0.5 * sigma**2 * Res[:,i - 1] * (dB[:,i-1]**2 - dt)

    return(Res)
    
    
Respuesta = intesto(mu,sigma,Te,n,tipo) #Tipo 1 es para Euler-Maruyama y 2 para Milstein