import numpy

"Parámetros  de entrada"
r=0.000097                 #Tasa libre de riesgo diaria
mu=r                       # se toma mu igual a la tasa LR, por ser neutrales al riesgo
sigma=0.0174               #Volatilidad diaria del rendimiento del activo
St=112                     #Precio Spot
Te=21                      #Tiempo AL vencimiento
metodo = 2                 #Método 1 es para Euler-Maruyama y 2 para Milstein
strike=110                 #precio de ejercicio de la opción
n=2000                     # Numero de simulaciones de MC

def numeric(mu,sigma,St,Te,metodo,n):
    dt = .01      # Tamaño de paso
    pasos = round(Te/dt) 
    dB = numpy.sqrt(dt) * numpy.random.normal(0,1,(n,pasos)) 
    solucion=numpy.zeros(shape=(n,pasos+1))
    solucion[:,0] = St # Condición Inicial
    
    if metodo==1:
        for i in range(1,pasos+1):
            solucion[:,i] = solucion[:,i - 1] + (mu * solucion[:,i - 1] * dt) + (sigma * solucion[:,i - 1] * dB[:,i-1])
    if metodo==2:
        for i in range(1,pasos+1):
            solucion[:,i] = solucion[:,i - 1] + mu * solucion[:,i - 1] * dt + sigma * solucion[:,i - 1] * dB[:,i-1] + 0.5 * sigma**2 * solucion[:,i - 1] * (dB[:,i-1]**2 - dt)

    return(solucion)
    
soluciones = numeric(mu,sigma,St,Te,metodo,n) 

Ct = soluciones[:,soluciones.shape[1]-1]-strike
Ct[Ct<0]=0
Ct=numpy.exp(-r*Te)*numpy.mean(Ct)

Pt = strike - soluciones[:,soluciones.shape[1]-1]
Pt[Pt<0]=0
Pt=numpy.exp(-r*Te)*numpy.mean(Pt)


