import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK

class GW: #onda gravitazionale che si propaga in direzione generica
    '''
    l'orientazione del cristalo è definita da un sistema di riferimento x, y, z;
    definisco un sistema di riferimento x', y', z' con z' orientato nella direzione di propagazione dell'onda:
    theta è l'angolo compreso tra z e z'
    phi è l'angolo compreso tra x e la proiezione di z' sul piano x-y;
    definisco ora un terzo sistema di riferimento x", y", z" con z" coincidente a z' e gli assi x" e y" allineati alle direzioni date da h+:
    psi è l'angolo compreso tra x" e x'
    '''
    def __init__(self): #basterà modificare il costruttore per aggiungere o togliere parametri necessari alle funzioni plus e cross
        #qui si inseriscono i parametri relativi all'onda
        self.theta = np.pi/2 #in radianti
        self.phi = 0 #in radianti
        self.psi = 0 #in radianti
        self.A = 10**-21 #ampiezza dell'onda gravitazionale (strain)
        self.omega = 3 #pulsazione dell'onda (MHz)

    def plus(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di h+ nel tempo (messa a caso per ora)
        return self.A*(self.omega**2)*np.cos(self.omega*t)

    def cross(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di hx nel tempo (messa a caso per ora)
        return self.A*(self.omega**2)*np.cos(self.omega*t)

    def h(self, t): #darivata seconda rispetto al tempo del tensore h
        p0 = self.plus(t) #ampiezza della polarizzazione plus nel sistema di riferimento "
        c0 = self.cross(t) #ampiezza della polarizzazione cross nel sistema di riferimento "

        #calcolo p e c nel sistema di riferimento '
        p = p0*np.cos(2*self.psi) - c0*np.sin(2*self.psi)
        c = p0*np.sin(2*self.psi) + c0*np.cos(2*self.psi)

        #definisco il sistema ' rispetto a quello del cristallo
        n=[np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)] #versore che definisce z'
        v=[-np.sin(self.phi), np.cos(self.phi), 0] #versore che definisce y'
        u=[np.cos(self.theta)*np.cos(self.phi), np.cos(self.theta)*np.sin(self.phi), -np.sin(self.theta)] #versore che definisce x'

        e_plus = [[0]*3 for _ in range(3)]
        e_cross = [[0]*3 for _ in range(3)]
        h = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                e_plus[i][j] = u[i]*u[j] - v[i]*v[j]
                e_cross[i][j] = u[i]*v[j] + v[i]*u[j]
                h[i][j] = p*e_plus[i][j] + c*e_cross[i][j]
        return h
    

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
    h = onda.h(t) #derivata seconda rispetto al tempo del tensore h
    #j=2 perchè solo le componenti con indice di colonna=z danno un risultato non nullo
    #i=2 per ora perchè non so quale direzione scegliere
    return (h[2][2]*xi)/2 - gamma*y - (omega**2)*x


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
