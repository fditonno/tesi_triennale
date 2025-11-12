#in questo provo i metodi numerici di risoluzione delle ODE applicandoli ad un oscillatore arminico con omega e gamma paragonabili a quelli del cristallo

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks
#from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK
#from equazioni_differenziali import Runge_Kutta_secondo_ordine_solve_ivp as RK
from equazioni_differenziali import Runge_Kutta_secondo_ordine_odeint as RK

def forzante(t):
    omega = 1e4 #pulsazione della forzante (KHz)
    periodo = 2*np.pi/omega
    ti = 2e-3 #tempo di inizio del segnale (ms)
    tf = 15*periodo + ti #tempo di fine del segnale (ms)
    if t<ti or t>tf:
        A=0
    else:
        A=1e5
    return A*np.sin(omega*(t-ti))
    
def f(t, x, y, omega, gamma): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
    return forzante(t) - gamma*y - (omega**2)*x

def scarto(t, x, y, omega, gamma): #calcolo lo scarto quadratico tra il lato sinistro e destro dell'equazione differenziale
    z = np.gradient(y, t) #derivata seconda di x rispetto al tempo
    
    scarto=[]
    somma=0
    normalizzazione = max(z[:(len(t)-1)]) #divido lo scarto per il massimo valore assuto da x" per far si che non dipenda dall'ampiezza scelta per la forzante
    for i in range(len(t)-1):
        scarto.append(((z[i] - f(t[i], x[i], y[i], omega, gamma))/normalizzazione)**2)
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto quadratico")
    ax.set_xlabel("Tempo (ms)")
    ax.plot(t[:(len(t)-1)], scarto, color = 'green')
    return somma/(len(t)-1)

def ampiezza(t, x): #funzione che calcola l'ampiezza della soluzione approssimata
    massimi, _ = find_peaks(x)
    minimi, _ = find_peaks(-np.array(x))

    tempi = []
    ampiezze = []
    for i in range(min([len(massimi), len(minimi)])):
        tempi.append(np.mean([t[massimi[i]], t[minimi[i]]]))
        ampiezze.append(x[massimi[i]] - x[minimi[i]])
    return tempi, ampiezze

def A(t, A0, gamma): #ampiezza della soluzione esatta (da dopo la fine del segnale)
    return A0*np.exp(-gamma*t/2)

#parametri per Runge Kutta
t_fine = 3e-2 #tempo di fine (ms)
t_inizio = 0 #tempo di inizio (ms)
N = 10000 #numero di punti che si vuole usare per l'approssimazione
B_0 = 0 #B(t_inzio)
dB_0 = 0 #B'(t_inzio)

#parametri oscillatore
omega = 1e4 #pulsazione (KHz)
gamma = 1e-3 #fattore di smorzamento (KHz)

#grafico della forzante
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Forzante")
ax.set_xlabel("Tempo (ms)")
x = np.linspace(t_fine, t_inizio, 10000)
y = []
for i in x:
    y.append(forzante(i))
ax.plot(x, y, color = 'red')

#grafico della soluzione
t, B, dB = RK(t_fine, t_inizio, N, B_0, dB_0, f, omega=omega, gamma=gamma)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione")
ax.set_xlabel("Tempo (ms)")
ax.plot(t, B, color = 'blue')

#grafico dell'ampiezza (da dopo la fine del segnale)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Ampiezza")
ax.set_xlabel("Tempo (ms)")
tempo, indice = 0.011425142514251425, t.index(0.011425142514251425) #considero la descrescita dell'ampiezza da dopo la fine del segnale
x, y = ampiezza(t[indice:], B[indice:])
ax.plot(x, y, color = 'blue', label='odeint')
x = np.linspace(t_fine, t[indice], 10000)
y = []
for i in x:
    y.append(2*A(i-tempo, max(B[indice:]), gamma))
ax.plot(x, y, color = 'red', label='soluzione esatta')
ax.legend()

print(scarto(t, B, dB, omega, gamma))

plt.show()
