import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK


class GW: #onda gravitazionale
    def __init__(self):
        #qui si inseriscono i parametri relativi all'onda
        self.a = (1/20)*10**-21 #coefficiente angolare della crescita dell'ampiezza dell'onda (strain/microS)
        self.b = 1/5 #coefficiente angolare della crescita della pulsazione dell'onda (MHz**2)

    def pulsazione(self, t): #funzione che descrive l'andamento della pulsazione dell'onda nel tempo
        #per ora ho messo una crescita lineare
        return self.b*t

    def ampiezza(self, t): #funzione che descrive l'andamento dell'ampiezza dell'onda nel tempo
        #per ora ho messo una crescita lineare
        return self.a*t

    def h(self, t): #funzione che descrive l'andamento di h nel tempo
        #considero h come uno scalare oscillante nel tempo dato che comunque servirà solamente una componenete
        omega = self.pulsazione(t)
        A = self.ampiezza(t)
        return A*np.cos(omega*t) #qualsiasi componente potrà sempre essere scritta così

    def ddh(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di h nel tempo
        omega = self.pulsazione(t)
        A = self.ampiezza(t)
        return A*(omega**2)*np.cos(omega*t) #per ora non ho considerato che A e omega dipendono dal tempo nella derivata
    

class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #qui si inseriscono i parametri relativi al cristallo
        self.d = 1 #spessore del cristallo (millimetri)
        self.L = 7 #raggio del cristallo (millimetri)
        self.R = 600 #raggio di curvatura della superficie del cristallo (millimetri)
        self.k = 10 #coefficiente piezzoelettrico (microC/mm) (è temporaneo)
        
        #qui si inseriscono i parametri relativi al modo di vibrazione scelto
        self.n = 1 #n del modo di vibrazione scelto (m=0, p=0)
        self.omega =  2 #pulsazione del modo di vibrazione scelto (MHz)
        self.gamma = 2*10**-7 #coefficente di smorzamento del modo scelto, ho stimato l'ordine di grandezza facendo omega/Q (MHz)
        self.chi_x = 250 #ho stimato l'ordine di grandezza da eta_x=10 (adimensionale)
        self.chi_y = 250 #ho stimato l'ordine di grandezza da eta_y=10 (adimensionale)

    def alfa(self): #equazione 7
        return self.chi_x/(np.pi*self.d*np.sqrt(self.R*self.L))

    def beta(self): #equazione 7
        return self.chi_y/(np.pi*self.d*np.sqrt(self.R*self.L))

    def eta_x(self): #trapping parameter asse x
        return self.L*np.sqrt(np.pi*self.alfa()/2)

    def eta_y(self): #trapping parameter asse y 
        return self.L*np.sqrt(np.pi*self.beta()/2)
        
    def xi(self): #parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        eta_x = self.eta_x()
        eta_y = self.eta_y()
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*eta_x)*erf(np.sqrt(self.n)*eta_y)
        denominatore = erf(np.sqrt(2*self.n)*eta_x)*erf(np.sqrt(2*self.n)*eta_y)
        return (self.d/2)*costante*numeratore/denominatore

    def corrente(self, dB): #funzione che restiruisce la corrente prodotta dal cristallo data la derivata rispetto al tempo del displacement (dB)
        #è temporanea
        I = []
        for i in dB:
            I.append(self.k*i)
        return I

    
def f(t, x, y, onda, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
    '''
    omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
    xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
    '''
    return (onda.ddh(t)*xi)/2 - gamma*y - (omega**2)*x


#parametri per Runge Kutta
t_fine = 20 #tempo di fine (miro secondi)
t_inizio = 0 #tempo di inizio (micro secondi)
N = 10000 #numero di punti che si vuole usare per l'approssimazione
B_0 = 0 #B(t_inzio) (millimetri)
dB_0 = 0 #B'(t_inzio) (millimetri/micro secondi)

onda = GW()
cristallo = BAW()
#calcolo il displacement (B) e la derivata rispetto al tempo di B (dB)
t, B, dB = RK(t_fine, t_inizio, (t_fine-t_inizio)/N, B_0, dB_0, f, onda=onda, omega=cristallo.omega, gamma=cristallo.gamma, xi=cristallo.xi())
I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo

#grafico della pulsazione dell'onda
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Pulsazione dell'onda")
ax.set_xlabel("Tempo (μs)")
ax.set_ylabel("Pulsazione (MHz)")
x = np.linspace(20, 0, 10000)
y = []
for i in x:
    y.append(onda.pulsazione(i))
ax.plot(x, y, label="onda", color = 'blue')
ax.axhline(y=cristallo.omega, label="cristallo", color='r', linestyle='--', linewidth=2)
ax.legend()

#grafico dell'onda
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Onda (h)")
ax.set_xlabel("Tempo (μs)")
ax.set_ylabel("Strain")
x = np.linspace(20, 0, 10000)
y = []
for i in x:
    y.append(onda.h(i))
ax.plot(x, y, color = 'blue')

#grafico del displacement
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Displacement (B)")
ax.set_xlabel("Tempo (μs)")
ax.set_ylabel("Displacement (mm)") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax.plot(t, B, color = 'blue')

#grafico della corrente
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Segnale")
ax.set_xlabel("Tempo (μs)")
ax.set_ylabel("Corrente (A)")
ax.plot(t, I, color = 'blue')

plt.show()
