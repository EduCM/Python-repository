import numpy
from pandas.io.data import DataReader
from datetime import datetime

act1 = DataReader('AC.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act1 = act1['Adj Close']
act2 = DataReader('ALSEA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act2 = act2['Adj Close']
act3 = DataReader('ASURB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act3 = act3['Adj Close']
act4 = DataReader('BIMBOA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act4 = act4['Adj Close']
act5 = DataReader('CEMEXCPO.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act5 = act5['Adj Close']
act6 = DataReader('ELEKTRA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act6 = act6['Adj Close']
act7 = DataReader('GAPB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act7 = act7['Adj Close']
act8 = DataReader('TLEVISACPO.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act8 = act8['Adj Close']
act9 = DataReader('GCARSOA1.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act9 = act9['Adj Close']
act10 = DataReader('GENTERA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act10 = act10['Adj Close']
act11 = DataReader('GFINBURO.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act11 = act11['Adj Close']
act12 = DataReader('GFNORTEO.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act12 = act12['Adj Close']
act13 = DataReader('GFREGIOO.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act13 = act13['Adj Close']
act14 = DataReader('ICA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act14 = act14['Adj Close']
act15 = DataReader('ICHB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act15 = act15['Adj Close']
act16 = DataReader('KIMBERA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act16 = act16['Adj Close']
act17 = DataReader('LABB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act17 = act17['Adj Close']
#act18 = DataReader('LACOMERUBC.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
#act18 = act18['Adj Close']
#act19 = DataReader('LALAB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
#act19 = act19['Adj Close']
act20 = DataReader('MEXCHEM.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act20 = act20['Adj Close']
#act21 = DataReader('NEMAKA.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
#act21 = act21['Adj Close']
act22 = DataReader('OHLMEX.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act22 = act22['Adj Close']
act23 = DataReader('OMAB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act23 = act23['Adj Close']
#act24 = DataReader('PE&OLES.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
#act24 = act24['Adj Close']
act25 = DataReader('SANMEXB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act25 = act25['Adj Close']
act26 = DataReader('SIMECB.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
act26 = act26['Adj Close']
#act27 = DataReader('SITESL.MX',  'yahoo', datetime(2013,1,3), datetime(2014,12,8))
#act27 = act27['Adj Close']

log1=numpy.log(act1.astype('float64')/act1.astype('float64').shift(1))
log2=numpy.log(act2.astype('float64')/act2.astype('float64').shift(1))
log3=numpy.log(act3.astype('float64')/act3.astype('float64').shift(1))
log4=numpy.log(act4.astype('float64')/act4.astype('float64').shift(1))
log5=numpy.log(act5.astype('float64')/act5.astype('float64').shift(1))
log6=numpy.log(act6.astype('float64')/act6.astype('float64').shift(1))
log7=numpy.log(act7.astype('float64')/act7.astype('float64').shift(1))
log8=numpy.log(act8.astype('float64')/act8.astype('float64').shift(1))
log9=numpy.log(act9.astype('float64')/act9.astype('float64').shift(1))
log10=numpy.log(act10.astype('float64')/act10.astype('float64').shift(1))
log11=numpy.log(act11.astype('float64')/act11.astype('float64').shift(1))
log12=numpy.log(act12.astype('float64')/act12.astype('float64').shift(1))
log13=numpy.log(act13.astype('float64')/act13.astype('float64').shift(1))
log14=numpy.log(act14.astype('float64')/act14.astype('float64').shift(1))
log15=numpy.log(act15.astype('float64')/act15.astype('float64').shift(1))
log16=numpy.log(act16.astype('float64')/act16.astype('float64').shift(1))
log17=numpy.log(act17.astype('float64')/act17.astype('float64').shift(1))
log18=numpy.log(act20.astype('float64')/act20.astype('float64').shift(1))
log19=numpy.log(act22.astype('float64')/act22.astype('float64').shift(1))
log20=numpy.log(act23.astype('float64')/act23.astype('float64').shift(1))
log21=numpy.log(act25.astype('float64')/act25.astype('float64').shift(1))
log22=numpy.log(act26.astype('float64')/act26.astype('float64').shift(1))

rendimientos=numpy.matrix([log1[2:len(log1)],log2[2:len(log2)],log3[2:len(log3)],log4[2:len(log4)],log5[1:len(log5)],log6[2:len(log6)],log7[2:len(log7)],log8[1:len(log8)],log9[2:len(log9)],log10[2:len(log10)],log11[2:len(log11)],log12[2:len(log12)],log13[2:len(log13)],log14[2:len(log14)],log15[2:len(log15)],log16[2:len(log16)],log17[2:len(log17)],log18[2:len(log18)],log19[2:len(log19)],log20[2:len(log20)],log21[2:len(log21)],log22[2:len(log22)]])

pesos=[0.01380,0.01280,0.01970,0.02260,0.05020,0.01150,0.02170,0.08510,0.00870,0.01350,0.02010,0.08530,0.00330,0.00090,0.00300,0.02150,0.00320,0.01340,0.00610,0.00700,0.01980,0.00110]
pesos=numpy.array(pesos)

rmean=rendimientos.mean(axis=1)

mediaportdiaria=rmean.transpose()@pesos
mediaportanual=mediaportdiaria*360

ract1=numpy.mean(log1[1:260])
ract2=numpy.mean(log2[1:260])
ract3=numpy.mean(log3[1:260])
ract4=numpy.mean(log4[1:260])
ract5=numpy.mean(log5[1:260])
ract6=numpy.mean(log6[1:260])
ract7=numpy.mean(log7[1:260])
ract9=numpy.mean(log9[1:260])
ract10=numpy.mean(log10[1:260])

vact1=numpy.std(log1[1:260])
vact2=numpy.std(log2[1:260])
vact3=numpy.std(log3[1:260])
vact4=numpy.std(log4[1:260])
vact5=numpy.std(log5[1:260])
vact6=numpy.std(log6[1:260])
vact7=numpy.std(log7[1:260])
vact9=numpy.std(log9[1:260])
vact10=numpy.std(log10[1:260])



rendport=[0]*260

for i in range(1,260):
    rendport[i-1]=log1[i]*.10+log2[i]*.10+log3[i]*.10+log4[i]*.10+log5[i]*.10+log6[i]*.10+log7[i]*.10+log9[i]*.10+log10[i]*.10
    

mediarendanual=numpy.mean(rendport)*360
desvportanual=numpy.std(rendport)*numpy.sqrt(360)
