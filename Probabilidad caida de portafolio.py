import numpy
from pandas.io.data import DataReader
from datetime import datetime

act1 = DataReader('ALFAA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
act1 = act1['Adj Close']
act2 = DataReader('PINFRA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
act2 = act2['Adj Close']
act3 = DataReader('ALPEKA.MX',  'yahoo', datetime(2015,2,26), datetime(2016,2,26))
act3 = act3['Adj Close']

def portafolioac(act1,act2,act3,w1,w2,w3,Te,n):
    
    ract1=numpy.log(act1.astype('float64')/act1.astype('float64').shift(1))
    ract2=numpy.log(act2.astype('float64')/act2.astype('float64').shift(1))
    ract3=numpy.log(act3.astype('float64')/act3.astype('float64').shift(1))

    vact1=numpy.std(numpy.log(act1.astype('float64')/act1.astype('float64').shift(1)))
    vact2=numpy.std(numpy.log(act2.astype('float64')/act2.astype('float64').shift(1)))
    vact3=numpy.std(numpy.log(act3.astype('float64')/act3.astype('float64').shift(1)))

    rend=numpy.matrix([ract1[1:len(ract1)],ract2[1:len(ract2)],ract3[1:len(ract3)]])
    
    matriz = numpy.cov(rend)*100
    chol = numpy.linalg.cholesky
    c = numpy.matrix(chol(matriz))
    z = numpy.random.normal(0, 1, size=(n,len(matriz)))
    x= c*z.transpose()
    x=x.transpose()

    stA=act1[len(act1)-1]
    stB=act2[len(act2)-1]
    stC=act3[len(act3)-1]

    STA=stA*numpy.exp((numpy.mean(rend[0,:])-(vact1**2)/2)*(Te)+(vact1*numpy.sqrt(Te)*(x[:,0])))
    STB=stB*numpy.exp((numpy.mean(rend[1,:])-(vact2**2)/2)*(Te)+(vact2*numpy.sqrt(Te)*(x[:,1])))
    STC=stC*numpy.exp((numpy.mean(rend[2,:])-(vact3**2)/2)*(Te)+(vact3*numpy.sqrt(Te)*(x[:,2])))

    po=w1*stA + w2*stB+stC*w3
    pT=w1*STA + w2*STB+STC*w3
    caida=(pT/po)
    caida=caida[caida<=1]
    return(caida.shape[1]/n)

prob=portafolioac(act1,act2,act3,.3,.4,.3,100,10000)