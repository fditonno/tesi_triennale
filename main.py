import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
#from numdifftools import Derivative
from equazioni_differenziali import Runge_Kutta_secondo_ordine_odeint as RK

#costanti fisiche
c = 3e8 #velocità della luce (m/s)
G = 6.67e-11 #costante di gravitazione universale (N*m^2/Kg^2) 

class GW: #onda gravitazionale
    def __init__(self):
        #parametri relativi all'onda
        self.ma = 2e28 #massa del primo oggetto (Kg) un po meno della massa del sole
        self.mb = 2e28 #massa del secondo oggetto (Kg) un po meno della massa del sole
        self.D = 1e19 #distanza dal sistema binario (m) dovrebbero essere circa 1000 anni luce
        self.T = 22e-6 #tempo in cui avviene la funzione degli oggetti (s)
        self.fase = 0 #fase (radianti)
        
        #parametri per il calcolo della derivata seconda
        self.passo = 1e-12 #è l'incremento che si sceglie di usare per il calcolo della derivata (s)

    def massa_chirp(self): #massa di chirp del sistema
        return ((self.ma*self.mb)**(3/5)) / ((self.ma+self.mb)**(1/5))

    def frequenza(self, t): #funzione che descrive la variazione della frequenza dell'onda nel tempo
        return ((G*self.massa_chirp()/(c**3))**(-5/8)) * ((1/(self.T-t))**(3/8))

    def ampiezza(self, t): #funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return (1/self.D) * ((G*self.massa_chirp()/(c**2))**(5/3)) * ((np.pi*self.frequenza(t)/c)**(2/3))
        
    def h(self, t): #funzione che descrive l'andamento di h nel tempo
        return self.ampiezza(t)*np.cos(2*np.pi*self.frequenza(t)*t + self.fase)

    def ddh(self, t): #derivata seconda rispetto al tempo della unzione che descrive l'andamento di h nel tempo
        '''
        ddh = Derivative(self.h, n=2) #derivata seconda di h
        return ddh(t)
        '''
        return (self.h(t+self.passo) - 2*self.h(t) + self.h(t-self.passo)) / (self.passo**2) #formula centrale per la derivata seconda
        
    
class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #parametri relativi al cristallo
        self.d = 1e-3 #spessore del cristallo (m)
        self.L = 14e-3 #raggio del cristallo (m)
        self.l = 7e-3 #raggio degli elettrodi (m)
        self.R = 600e-3 #raggio di curvatura della superficie del cristallo (m)
        
        #parametri relativi al modo di vibrazione scelto
        self.n = 3 #n del modo di vibrazione scelto (m=0, p=0, n dispari)
        self.omega = 2*np.pi*5e6 #pulsazione del modo di vibrazione scelto (Hz)
        self.gamma = 2*np.pi*5e6/1e7 #coefficente di smorzamento del modo scelto (Hz)
        self.eta_x = 1 #trapping parameter asse x (adimensionale)
        self.eta_y = 1 #trapping parameter asse y (adimensionale)
        self.k = 1e-2 #coefficiente di accoppiamento elettromeccanico (C/m)
            
    def xi(self): #parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*self.eta_x)*erf(np.sqrt(self.n)*self.eta_y)
        denominatore = erf(np.sqrt(2*self.n)*self.eta_x)*erf(np.sqrt(2*self.n)*self.eta_y)
        return (self.d/2)*costante*numeratore/denominatore

    def alfa(self):
        return (2/np.pi)*(self.eta_x/self.L)**2

    def beta(self):
        return (2/np.pi)*(self.eta_y/self.L)**2

    def Ux(self, x): #dipendenza da x della parte spaziale del displacement (m=p=0)
        return np.exp(-np.pi*self.n*self.alfa()*(x**2)/2)

    def Uy(self, y): #dipendenza da y della parte spaziale del displacement (m=p=0)
        return np.exp(-np.pi*self.n*self.beta()*(y**2)/2)
    
    def corrente(self, dB): #funzione che restiruisce la corrente prodotta dal cristallo data la derivata rispetto al tempo della parte temporale displacement (dB)
        integrale_x, _ = quad(self.Ux, -self.l, self.l) #integrale di Ux tra -l e l
        integrale_y, _ = quad(self.Uy, -self.l, self.l) #integrale di Uy tra -l e l
        #cost = self.k * integrale_x * integrale_y
        cost = self.k
        I = []
        for i in dB:
            I.append(cost*i)
        return I


def risolvi(t_fine, t_inizio, N, B_0, dB_0, onda, cristallo): #funzione per la risoluzione dell'equazione differenziale
    '''
    per far si che RK funzioni bene è necessario che omega e gamma non siano numeri troppo grandi o troppo piccoli per questo converto tutto in KHz;
    inioltre l'ampiezza della forzante non deve essere troppo piccola quindi moltiplico da entrambi i lati dell'equazione per un fattore di conversione
    che rende l'ampiezza della forzante un numero dell'ordine di grandezza di 1e5:
    forzante(t)*conv = (ddB*conv) + (dB*conv)*gamma + (B*conv)*omega^2 = ddb + db*gamma + b*omega^2
    dopo aver applicato RK riconverto tutte le unità di misura a SI e divido b e db per il fattore di conversione in modo da riottenere B e dB.
    '''
    t_fine = t_fine*1e3 #passo da s a ms
    t_inizio = t_inizio*1e3 #passo da s a ms
    dB = dB_0*1e-3 #passo da m/s a m/ms
    
    massimo=-1 #massimo valore assunto da ddh
    for i in np.linspace(t_fine, t_inizio, N):
        ddh = onda.ddh(i*1e-3) #ddh riceve in input s e non ms quindi torno a SI
        ddh = ddh*1e-6 #passo da strain/s^2 a strain/ms^2
        if ddh>massimo:
            massimo=ddh
    xi=cristallo.xi()
    ordine=1e5
    conv = 2*ordine/(massimo*xi) #fattore che converte il termine (onda.ddh(t)*xi)/2 ad un numero dell'ordine di grandezza di "ordine"
    
    def f(t, x, y, onda, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
        '''
        omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
        xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
        '''
        return conv*(onda.ddh(t*1e-3)*1e-6*xi)/2 - gamma*y - (omega**2)*x #f misurata in m/ms^2

    #applico RK                                                                 passo da Hz a KHz
    t, b, db = RK(t_fine, t_inizio, N, B_0, dB_0, f, onda=onda, omega=cristallo.omega*1e-3, gamma=cristallo.gamma*1e-3, xi=xi) #b=B*conv e db=dB*conv

    #ritorno alle unità di misura iniziali
    t_SI=[]
    B=[]
    dB=[]
    for i in range(len(t)):
        t_SI.append(t[i]*1e-3) #passo da ms a s
        B.append(b[i]/conv) #divido per conv per convertire b in B
        dB.append(db[i]*1e3/conv) #passo da m/ms a m/s e divido per conv per convertire db in dB
    return t_SI, B, dB

def scarto(t, x, y, onda, omega, gamma, xi): #calcolo lo scarto quadratico tra x" calcolato numericamente a partire da x e f(t, x, x') per valutare la bontà dell'approssimazione
    z = np.gradient(y, t) #derivata seconda di x rispetto al tempo

    def f(t, x, y, onda, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
        '''
        omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
        xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
        '''
        return (onda.ddh(t)*xi)/2 - gamma*y - (omega**2)*x #misurato in SI
    
    normalizzazione = max(z[:(len(t)-1)]) #divido lo scarto per il massimo valore assuto da x" per far si che non dipenda dall'ampiezza scelta per la forzante
    scarto=[]
    somma=0
    for i in range(len(t)-1):
        scarto.append(((f(t[i], x[i], y[i], onda, omega, gamma, xi)-z[i])/normalizzazione)**2)
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto quadratico")
    ax.set_xlabel("Tempo (ms)")
    ax.plot(t[:(len(t)-1)], scarto, color = 'green')
    return somma/(len(t)-1)

#parametri per Runge Kutta
t_fine = 18e-6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
N = 10000 #numero di punti che si vuole usare per l'approssimazione
B_0 = 0 #B(t_inzio) (millimetri)
dB_0 = 0 #B'(t_inzio) (millimetri/micro secondi)

onda = GW()
cristallo = BAW()
t, B, dB = risolvi(t_fine, t_inizio, N, B_0, dB_0, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo

#grafico dell'onda
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Onda (h)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Strain")
x = np.linspace(t_fine, t_inizio, 10000)
y = []
for i in x:
    y.append(onda.h(i))
ax.plot(x, y, color = 'blue')

#grafico del displacement
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("B")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Displacement (m)") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax.plot(t, B, color = 'blue')

#grafico della corrente
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Segnale")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Corrente (A)")
ax.plot(t, I, color = 'blue')

print(scarto(t, B, dB, onda, cristallo.omega, cristallo.gamma, cristallo.xi()))

plt.show()
