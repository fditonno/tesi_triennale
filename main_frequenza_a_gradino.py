#in questo file simulo il segnale generato da un'onda gravitazionale ad ampiezza costante e frequenza che crese come un gradino per poter utilizzare la soluzione analitica

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import equazione_differenziale as ode

class GW: #onda gravitazionale
    def __init__(self):
        #parametri relativi all'onda
        self.inizio = [0, 5e-6, 10e-6, 15e-6, 20e-6, 25e-6, 30e-6, 35e-6] #tempi di inizio dei segnali monocromatici (s)
        #self.pulsazione = [2*np.pi*4.6e6, 2*np.pi*4.8e6, 2*np.pi*4.9e6, -2*np.pi*5e6, -2*np.pi*5.1e6, 2*np.pi*5.2e6, 2*np.pi*5.4e6] #pulsazioni (Hz)
        self.pulsazione = [2*np.pi*5.4e6, 2*np.pi*5.2e6, 2*np.pi*5.1e6, -2*np.pi*5e6, -2*np.pi*4.9e6, 2*np.pi*4.8e6, 2*np.pi*4.6e6] #pulsazioni (Hz)
        self.ampiezza = [1e-21, 1e-21, 1e-21, 1e-21, 1e-21, 1e-21, 1e-21] #ampiezze (strain)
        self.fase = -np.pi/2 #fase (radianti)

    def h(self, t): #funzione che descrive l'andamento di h nel tempo
        for j in range(len(onda.inizio)-1):
            if t<onda.inizio[j+1] and t>=onda.inizio[j]: #capisco quale onda monocromatica usare
                indice = j
        return self.ampiezza[indice] * np.cos(self.pulsazione[indice]*t + self.fase)

    def dd_h(self, t): #derivata seconda della unzione che descrive l'andamento di h nel tempo
        for j in range(len(onda.inizio)-1):
            if t<onda.inizio[j+1] and t>=onda.inizio[j]: #capisco quale onda monocromatica usare
                indice = j
        return -(self.pulsazione[indice]**2) * self.ampiezza[indice] * np.cos(self.pulsazione[indice]*t + self.fase) 


class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #parametri relativi al cristallo
        self.d = 1e-3 #spessore del cristallo (m)
        
        #parametri relativi al modo di vibrazione
        self.n = 3 #n del modo di vibrazione (m=0, p=0, n dispari)
        self.omega = 2*np.pi*5e6 #pulsazione del modo di vibrazione (Hz)
        self.gamma = 2*np.pi*5e6/1e7 #coefficente di smorzamento del modo di vibrazione(Hz)
        self.eta_x = 0.1 #trapping parameter asse x (adimensionale)
        self.eta_y = 0.1 #trapping parameter asse y (adimensionale)
        self.k = 1e-2 #coefficiente di accoppiamento elettromeccanico (C/m)
            
    def xi(self): #parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*self.eta_x)*erf(np.sqrt(self.n)*self.eta_y)
        denominatore = erf(np.sqrt(2*self.n)*self.eta_x)*erf(np.sqrt(2*self.n)*self.eta_y)
        return (self.d/2)*costante*numeratore/denominatore
    
    def corrente(self, dB): #funzione che restiruisce la corrente prodotta dal cristallo data la derivata rispetto al tempo della parte temporale displacement (dB)
        I = []
        for i in dB:
            I.append(self.k*i)
        return I


#valori iniziali
t_fine = 35e-6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
B_0 = 0 #B(0) (m)
dB_0 = 0 #B'(0) (m/s)

#parametri per Runge Kutta
N=100000 #numero di punti in cui si vuole la soluzione

onda = GW()
cristallo = BAW()
#uso la soluzione analitica perchè in questo caso esiste
t, B, dB = ode.risolvi_analitica(t_fine, t_inizio, 0, B_0, dB_0, N, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo


#grafici
fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex=True)
ax[2].set_xlabel("Tempo [s]")

#grafico della frequenza dell'onda
ax[0].set_ylabel("Frequenza [Hz]")
ax[0].grid(True)
for i in range(len(onda.pulsazione)):
    ax[0].hlines(y=abs(onda.pulsazione[i])/(2*np.pi), xmin=onda.inizio[i], xmax=onda.inizio[i+1], color = 'green')
    if abs(onda.pulsazione[i]) == cristallo.omega:
        x_min = onda.inizio[i]
        x_max = onda.inizio[i+1]
ax[0].axvspan(x_min, x_max, alpha=0.4, color = 'red', label="risonanza") #area relativa alla frequenza di risonanza
ax[0].legend(loc='upper left')

#grafico del displacement
ax[1].set_ylabel("Displacement [m]") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax[1].grid(True)
ax[1].plot(t, B, color = 'blue')
ax[1].axvspan(x_min, x_max, alpha=0.4, color = 'red', label="risonanza") #area relativa alla frequenza di risonanza
ax[1].legend(loc='upper left')

#grafico della corrente
ax[2].set_ylabel("Corrente [A]")
ax[2].grid(True)
ax[2].plot(t, I, color = 'blue')
ax[2].axvspan(x_min, x_max, alpha=0.4, color = 'red', label="risonanza") #area relativa alla frequenza di risonanza
ax[2].legend(loc='upper left')

plt.show()
