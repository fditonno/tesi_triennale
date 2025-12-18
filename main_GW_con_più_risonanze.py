#in questo file simulo il segnale generato dalla fusione di 2 buchi neri con una massa di 4e-3 masse solari ad una distanza di 1e5 anni luce

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.signal import spectrogram
#from matplotlib.colors import LogNorm
from scipy.fft import fft, fftfreq
import equazione_differenziale as ode

#costanti fisiche
c = 299792458 #velocità della luce (m/s)
G = 6.6743e-11 #costante di gravitazione universale (N*m^2/Kg^2)
ly = 9.461e15 #anno luce (m)
M_sol = 1.989e30 #massa del sole (Kg)

class GW: #onda gravitazionale
    def __init__(self, ma, mb, D, T, fase_iniziale):
        self.ma = ma
        self.mb = mb
        self.D = D
        self.T = T
        self.fase_iniziale = fase_iniziale

        #calcolo le costanti che compaiono nelle funzioni per evitare di ricalcolarle inutilmente
        self.massa_chirp = ((self.ma*self.mb)**(3/5)) / ((self.ma+self.mb)**(1/5)) #massa di chirp del sistema
        self.cost_f = (G*self.massa_chirp/(c**3))**(-5/8) #costante che compare nella funzione frequenza e nelle sue derivate
        self.cost_A = (1/self.D) * ((G*self.massa_chirp/(c**2))**(5/3)) * ((np.pi/c)**(2/3)) #costante che compare nella funzione ampiezza e nelle sue derivate

    def frequenza(self, t): #funzione che descrive la variazione della frequenza dell'onda nel tempo
        return self.cost_f * ((self.T-t)**(-3/8))

    def d_frequenza(self, t): #derivata prima della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return (3/8) * self.cost_f * ((self.T - t)**(-11/8))

    def dd_frequenza(self, t): #derivata seconda della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return (33/64) * self.cost_f * ((self.T - t)**(-19/8))

    def ampiezza(self, t): #funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return self.cost_A * (self.frequenza(t)**(2/3))

    def d_ampiezza(self, t): #derivata prima della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return (2/3) * self.cost_A * (self.frequenza(t)**(-1/3)) * self.d_frequenza(t)

    def dd_ampiezza(self, t): #derivata seconda della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return (2/3)*self.cost_A * (((-1/3) * (self.frequenza(t)**(-4/3)) * (self.d_frequenza(t)**2)) + (self.frequenza(t)**(-1/3) * self.dd_frequenza(t)))

    def fase(self, t, t_inizio): #fase al tempo t, ovvero integrale tra t_inizio e t in dtau di 2*np.pi*frequenza(tau) + fase_iniziale
        def integrale(tau):
            return (2*np.pi) * (-8/5) * self.cost_f * ((self.T-tau)**(5/8))
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
    def __init__(self, d, n, omega, gamma, eta_x, eta_y, k):
        self.d = d
        self.n = n
        self.omega = omega
        self.gamma = gamma
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.k = k
        
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


#parametri dell'onda gravitazionale
ma = 2.2e-3 * M_sol #massa del primo oggetto (Kg)
mb = 2.2e-3 * M_sol #massa del secondo oggetto (Kg)
D = 1e5 * ly #distanza dal sistema binario (m)
T = 36e-6 #tempo in cui avviene la funzione degli oggetti (s)
fase_iniziale = 0 #fase a t=t_inizio (radianti)

#parametri del BAW
d = 1e-3 #spessore del cristallo (m)
n = [3, 5, 7] #n dei modi di vibrazione (m=0, p=0, n dispari)
omega = [2*np.pi*5.168e6, 2*np.pi*8.613e6, 2*np.pi*1.206e7] #pulsazioni dei modi di vibrazione (Hz)
Q = [6.81e6, 1.81e7, 2.53e7] #quality factor dei modi di vibrazione
eta_x = 0.1 #trapping parameter asse x (adimensionale)
eta_y = 0.1 #trapping parameter asse y (adimensionale)
k = 1e-2 #coefficiente di accoppiamento elettromeccanico (C/m)

#valori iniziali
t_fine = 35e-6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
B_0 = 0 #B(t_inzio) (m)
dB_0 = 0 #B'(t_inzio) (m/s)

#parametri per Runge Kutta
N = 1000000 #numero di punti in cui si vuole la soluzione, non influisce sulla precisione di RK ma serve un numero alto per far si che lo scarto sia realistico
rtol = 1e-8 #errore relativo massimo che si vuole sui punti della soluzione (1e-8 è il valore che sembra funzionare meglio)
atol = rtol*1e-24 #errore assoluto massimo che si vuole sui punti della soluzione (m)

onda = GW(ma, mb, D, T, fase_iniziale)
B = np.zeros(N)
I = np.zeros(N)
scarto = np.zeros(N)
somma_scarto = 0
cont = False
for i in range(len(n)):
    #calcolo B e I
    cristallo = BAW(d, n[i], omega[i], omega[i]/Q[i], eta_x, eta_y, k)
    t, B_i, dB_i = ode.risolvi_RK(t_fine, t_inizio, B_0, dB_0, N, rtol, atol, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
    I_i = cristallo.corrente(dB_i) #calcolo la corrente prodotta dal cristallo
    scarto_i, somma_scarto_i = ode.scarto(t, B_i, dB_i, onda, cristallo) #lo scarto è circa sovrastimato di un'ordine di grandezza usando N=1000000
    
    #sommo le grandezze punto per punto
    for j in range(N):
        B[j] = B[j] + B_i[j]
        I[j] = I[j] + I_i[j]
        scarto[j] = np.sqrt(scarto[j]**2 + scarto_i[j]**2)
    somma_scarto = np.sqrt(somma_scarto**2 + somma_scarto_i**2)


#grafici dell'onda gravitazioale
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex=True)
x = np.linspace(t_inizio, t_fine, N)
ax[1].set_xlabel("t - t_merging [s]")

#grafico di h
ax[1].set_ylabel("h [strain]")
ax[1].grid(True)
y = []
for i in x:
    y.append(onda.h(i, t_inizio))
ax[1].plot(x-onda.T, y, color = 'green')

#grafico della frequenza
ax[0].set_ylabel("Frequenza [Hz]")
ax[0].grid(True)
y = []
for i in x:
    y.append(onda.frequenza(i))
ax[0].plot(x-onda.T, y, color = 'green')

#linee che corrispondono alle frequenze di risonanza
x_linee = []
for omega_n in omega:
    i=0
    while(2*np.pi*y[i] < omega_n):
        indice_linea = i
        i = i+1
    x_linee.append(np.mean([x[indice_linea], x[indice_linea+1]]))
first = True
for x_linea in x_linee:
    if first:
        ax[0].axvline(x_linea-onda.T, linestyle='--', color = 'red', label="risonanze")
        ax[1].axvline(x_linea-onda.T, linestyle='--', color = 'red', label="risonanze")
        first = False
    else:
        ax[0].axvline(x_linea-onda.T, linestyle='--', color = 'red')
        ax[1].axvline(x_linea-onda.T, linestyle='--', color = 'red')
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')


#grafici del segnale
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex=True)
ax[1].set_xlabel("t - t_merging [s]")

#grafico del displacement
ax[0].set_ylabel("Displacement [m]") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax[0].grid(True)
ax[0].plot(t-onda.T, B, color = 'blue')
first = True
for x_linea in x_linee:
    if first:
        ax[0].axvline(x_linea-onda.T, linestyle='--', color = 'red', label="risonanze")
        first = False
    else:
        ax[0].axvline(x_linea-onda.T, linestyle='--', color = 'red')
ax[0].legend(loc='upper left')

#grafico della corrente
ax[1].set_ylabel("Corrente [A]")
ax[1].grid(True)
ax[1].plot(t-onda.T, I, color = 'blue')
first = True
for x_linea in x_linee:
    if first:
        ax[1].axvline(x_linea-onda.T, linestyle='--', color = 'red', label="risonanze")
        first = False
    else:
        ax[1].axvline(x_linea-onda.T, linestyle='--', color = 'red')
ax[1].legend(loc='upper left')

#spettrogramma
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_xlabel("t - t_merging [s]")
ax.grid(True)
frequenza, tempi, Sxx = spectrogram(np.array(I), N/(t_fine-t_inizio), scaling='spectrum', mode='magnitude', window = 'hann', nperseg=32768, nfft = 32768, noverlap=32768*8//10)
ax.set_ylabel('Frequenza [Hz]')
ax.set_xlim([t_inizio-onda.T, t_fine-onda.T])
ax.set_ylim([0, 1.5e7])
pcm = ax.pcolormesh(tempi-onda.T, frequenza, Sxx, shading='gouraud') #norm=LogNorm()
fig.colorbar(pcm, ax=ax, label='Ampiezza [A]')
first = True
for x_linea in x_linee:
    if first:
        ax.axvline(x_linea-onda.T, linestyle='--', color = 'red', label="risonanze")
        first = False
    else:
        ax.axvline(x_linea-onda.T, linestyle='--', color = 'red')
ax.plot(x-onda.T, y, color = 'green', label="frequenza dell'onda")
ax.legend(loc='upper left')


#grafico dello scarto
print(somma_scarto)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_xlabel("t - t_merging [s]")
ax.set_yscale("log")
ax.set_ylabel("Scarto relativo")
ax.plot(t-onda.T, scarto, color = 'orange')

plt.show()
