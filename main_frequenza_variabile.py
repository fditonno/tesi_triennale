#in questo file simulo il segnale generato da un'onda gravitazionale ad ampiezza costante e frequenza variabile

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.signal import spectrogram
import equazione_differenziale as ode

class GW: #onda gravitazionale
    def __init__(self):
        #parametri relativi all'onda
        self.m = 0.5e6/3e-5
        self.q = 5e6 - 2.5e5
        self.fase_iniziale = 0 #fase al tempo iniziale (radianti)

    def frequenza(self, t): #funzione che descrive la variazione della frequenza dell'onda nel tempo
        return self.m*t + self.q

    def d_frequenza(self, t): #derivata prima della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return self.m

    def dd_frequenza(self, t): #derivata seconda della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return 0

    def ampiezza(self, t): #funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return 1e-21 #ampiezza (strain)

    def d_ampiezza(self, t): #derivata prima della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return 0

    def dd_ampiezza(self, t): #derivata seconda della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return 0

    def fase(self, t, t_inizio): #fase al tempo t, ovvero integrale tra t_inizio e t in dtau di 2*np.pi*frequenza(tau) + fase_iniziale
        def integrale(tau):
            return (2*np.pi) * ((self.m*(tau**2)/2) + (self.q*tau)) 
        return integrale(t) - integrale(t_inizio) + self.fase_iniziale

    def h(self, t, t_inizio): #funzione che descrive l'andamento di h nel tempo
        return self.ampiezza(t)*np.cos(self.fase(t, t_inizio))

    def dd_h(self, t, t_inizio): #derivata seconda della unzione che descrive l'andamento di h nel tempo
        A = self.ampiezza(t)
        d_A = self.d_ampiezza(t)
        dd_A = self.dd_ampiezza(t)
        phi = self.fase(t, t_inizio)
        d_phi = 2*np.pi*self.frequenza(t) #dato che la fase è l'integrale della frequenza
        dd_phi =  2*np.pi*self.d_frequenza(t)
        return dd_A*np.cos(phi) - 2*d_A*d_phi*np.sin(phi) - A*((d_phi)**2)*np.cos(phi) - A*dd_phi*np.sin(phi)

 
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
t_fine = 30e-6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
B_0 = 0 #B(t_inzio) (m)
dB_0 = 0 #B'(t_inzio) (m/s)

#parametri per Runge Kutta
N = 1000000 #numero di punti in cui si vuole la soluzione, non influisce sulla precisione di RK ma serve un numero alto per far si che lo scarto sia realistico
rtol = 1e-8 #errore relativo massimo che si vuole sui punti della soluzione (1e-8 è il valore che sembra funzionare meglio)
atol = rtol*1e-24 #errore assoluto massimo che si vuole sui punti della soluzione

#calcolo B e I
onda = GW()
cristallo = BAW()
t, B, dB = ode.risolvi_RK(t_fine, t_inizio, B_0, dB_0, N, rtol, atol, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo


#grafici
fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex=True)
x = np.linspace(t_inizio, t_fine, N)
ax[2].set_xlabel("Tempo [s]")

#grafico della frequenza
ax[0].set_ylabel("Frequenza [Hz]")
ax[0].grid(True)
y = []
for i in x:
    y.append(onda.frequenza(i))
ax[0].plot(x, y, color = 'green')

#linea che corrisponde alla frequenza di risonanza
i=0
while(2*np.pi*y[i] < cristallo.omega):
    indice_linea = i
    i = i+1
x_linea = np.mean([x[indice_linea], x[indice_linea+1]])
ax[0].axvline(x_linea, linestyle='--', color = 'red', label="risonanza")
ax[0].legend(loc='upper left')

#grafico del displacement
ax[1].set_ylabel("Displacement [m]") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax[1].grid(True)
ax[1].plot(t, B, color = 'blue')
ax[1].axvline(x_linea, linestyle='--', color = 'red', label="risonanza")
ax[1].legend(loc='upper left')

#grafico della corrente
ax[2].set_ylabel("Corrente [A]")
ax[2].grid(True)
ax[2].plot(t, I, color = 'blue')
ax[2].axvline(x_linea, linestyle='--', color = 'red', label="risonanza")
ax[2].legend(loc='upper left')


#grafico dello scarto
y, scarto = ode.scarto(t, B, dB, onda, cristallo) #lo scarto è circa sovrastimato di un'ordine di grandezza usando N=1000000
print(scarto)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_xlabel("Tempo [s]")
ax.set_yscale("log")
ax.set_ylabel("Scarto relativo")
ax.plot(t, y, color = 'orange')

plt.show()
