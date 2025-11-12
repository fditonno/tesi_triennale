#in questo provo i metodi numerici di risoluzione delle ODE applicandoli ad un oscillatore arminico con omega e gamma paragonabili a quelli del cristallo

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks
#from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK
#from equazioni_differenziali import Runge_Kutta_secondo_ordine_solve_ivp as RK
from equazioni_differenziali import Runge_Kutta_secondo_ordine_odeint as RK

def forzante(t):
    return 0
    
def f(t, x, y, omega, gamma): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
    return forzante(t) - gamma*y - (omega**2)*x

def ampiezza(t, x): #funzione che calcola l'ampiezza della soluzione approssimata
    massimi, _ = find_peaks(x)
    minimi, _ = find_peaks(-np.array(x))

    tempi = []
    ampiezze = []
    for i in range(min([len(massimi), len(minimi)])):
        tempi.append(np.mean([t[massimi[i]], t[minimi[i]]]))
        ampiezze.append(x[massimi[i]] - x[minimi[i]])
    return tempi, ampiezze

def A(t, A0, gamma): #ampiezza della soluzione esatta
    return A0*np.exp(-gamma*t/2)

def soluzione(t, A0, omega, gamma): #soluzione esatta dell'equazione differenziale
    pulsazzione = np.sqrt(4*(omega**2) - (gamma**2))/2
    return A(t, A0, gamma)*np.cos(pulsazzione*t)

def scarto(t, x, **parametri): #scarto quadratico tra la soluzione e l'approssimazione
    scarto=[]
    somma=0
    normalizzazione = max(x) #divido lo scarto per il massimo valore assuto dalla soluzione per far si che non dipenda dall'ampiezza scelta per la forzante
    for i in range(len(t)):
        scarto.append(((soluzione(t[i], **parametri)-x[i])/normalizzazione)**2)
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto quadratico")
    ax.set_xlabel("Tempo (ms)")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t)


#parametri per Runge Kutta
t_fine = 3e-2 #tempo di fine (ms)
t_inizio = 0 #tempo di inizio (ms)
N = 10000 #numero di punti che si vuole usare per l'approssimazione
B_0 = 1 #B(t_inzio)
dB_0 = 0 #B'(t_inzio)

#parametri oscillatore
omega = 1e4 #pulsazione (KHz)
gamma = 1e-3 #fattore di smorzamento (KHz)

#grafico della soluzione
t, B, dB = RK(t_fine, t_inizio, N, B_0, dB_0, f, omega=omega, gamma=gamma)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione")
ax.set_xlabel("Tempo (ms)")
ax.plot(t, B, color = 'blue')

#grafico dell'ampiezza
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Ampiezza")
ax.set_xlabel("Tempo (ms)")
x, y = ampiezza(t, B)
ax.plot(x, y, color = 'blue', label='odeint')
x = np.linspace(t_fine, t_inizio, 10000)
y = []
for i in x:
    y.append(2*A(i, B_0, gamma))
ax.plot(x, y, color = 'red', label='soluzione esatta')
ax.legend()

print(scarto(t, B, A0=B_0, omega=omega, gamma=gamma))

plt.show()
