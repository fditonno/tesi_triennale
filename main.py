import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK


class GW: #onda gravitazionale
    def __init__(self):
        #qui si inseriscono i parametri relativi all'onda
        self.A = 10**-21 #ampiezza dell'onda gravitazionale (strain)
        self.omega = 3 #pulsazione dell'onda (MHz)

    def h(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di h nel tempo
        #considero h come uno scalare oscillante nel tempo dato che comunque servirà solamente una componenete
        return self.A*(self.omega**2)*np.cos(self.omega*t) #qualsiasi componente potrà sempre essere scritta così
    

class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #qui si inseriscono i parametri relativi al cristallo e al modo di vibrazione
        self.d = 1 #spessore del cristallo (millimetri)
        self.n = 3 #n del modo di vibrazione scelto (m=0, p=0)
        self.omega =  5.175 #pulsazione del modo di vibrazione scelto (MHz)
        self.gamma = 10**-8 #coefficente di smorzamento del modo scelto (MHz)
        self.eta_x = 10 #trapping parameter per l'asse x
        self.eta_y = 10 #trapping parameter per l'asse y

    def xi(self): #calcolo del parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*self.eta_x)*erf(np.sqrt(self.n)*self.eta_y)
        denominatore = erf(np.sqrt(2*self.n)*self.eta_x)*erf(np.sqrt(2*self.n)*self.eta_y)
        return (self.d/2)*costante*numeratore/denominatore

    
def f(t, x, y, onda, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
    '''
    omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
    xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
    '''
    return (onda.h(t)*xi)/2 - gamma*y - (omega**2)*x


#parametri per Runge Kutta
t_fine = 20 #tempo di fine (miro secondi)
t_inizio = 0 #tempo di inizio (micro secondi)
N = 10000 #numero di punti che si vuole usare per l'approssimazione
x0 = 0 #B(t_inzio) (millimetri)
y0 = 0 #B'(t_inzio) (millimetri)

onda = GW()
cristallo = BAW()
t, B, _ = RK(t_fine, t_inizio, (t_fine-t_inizio)/N, x0, y0, f, onda=onda, omega=cristallo.omega, gamma=cristallo.gamma, xi=cristallo.xi())

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_xlabel("μs")
ax.set_ylabel("mm") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax.plot(t, B, label="B(t)", color = 'blue')
ax.legend()
plt.show()
