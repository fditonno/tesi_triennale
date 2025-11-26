#in questo file applico RK alla risoluzione dell'equazione dell'oscillatore smorzato

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

def risolvi(t_fine, t_inizio, N, x_0, dx_0, omega, gamma): #funzione per la risoluzione dell'equazione differenziale
    def sistema(t, v, omega, gamma): #sistema di ODE da risolvere
        x, y = v
        
        def f(t, x, y, omega, gamma): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
            return - gamma*y - (omega**2)*x
        
        dx_dt = y #x' = g(t, x, y) = y
        dy_dt = f(t, x, y, omega, gamma) #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, omega, gamma): #matrice jacobiana del sistema
        '''
        [[dg/dx, dg/dy],
        [df/dx, df/dy]]
        '''
        return [[0, 1],
                [-(omega**2), -gamma]]

    sol = solve_ivp(sistema,
                    [t_inizio, t_fine],
                    [x_0, dx_0], #condizioni iniziali x_0 = x(t_inizio), dx_0 = x'(t_inizio)
                    method = 'Radau', #metodi: RK45, DOP853, Radau, BDF
                    t_eval = np.linspace(t_inizio, t_fine, N), #punti in cui si vuole la soluzione
                    args = (omega, gamma), #parametri di f
                    first_step = (t_fine-t_inizio)/1e4, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=1e-8, #errore relativo massimo accettabile
                    atol=1e-8*x_0, #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t.tolist(), x.tolist(), y.tolist() 

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

def scarto(t, x, **parametri): #scarto relativo tra la soluzione e l'approssimazione
    scarto=[]
    somma=0
    for i in range(len(t)):
        sol = soluzione(t[i], **parametri)
        scarto.append(np.abs((sol-x[i])/sol))
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto relativo")
    ax.set_xlabel("Tempo (s)")
    ax.set_yscale("log")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t) #divido la somma per il numero di punti per normalizzare


#parametri per Runge Kutta
t_fine = 30e-5 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
N = 300000 #numero di punti in cui si vuole la soluzione
x_0 = 1e-21 #x(t_inzio) (l'ho dell'ordine di grandezza giusto per avere x della stessa dimensione di quello del caso con l'onda gravitazionale)
dx_0 = 0 #x'(t_inzio)

#parametri oscillatore
omega = 1e7 #pulsazione (Hz)
gamma = 1 #fattore di smorzamento (Hz)

#grafico della soluzione
t, x, dx = risolvi(t_fine, t_inizio, N, x_0, dx_0, omega, gamma)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione")
ax.set_xlabel("Tempo (s)")
ax.plot(t, x, color = 'blue', label='RK')
asse_x = np.linspace(t_fine, t_inizio, N)
asse_y = []
for i in asse_x:
    asse_y.append(soluzione(i, x_0, omega, gamma))
ax.plot(asse_x, asse_y, color = 'red', label='analitica')
ax.legend()

#grafico dell'ampiezza
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Ampiezza")
ax.set_xlabel("Tempo (s)")
asse_x, asse_y = ampiezza(t, x)
ax.plot(asse_x, asse_y, color = 'blue', label='RK')
asse_x = np.linspace(t_fine, t_inizio, N)
asse_y = []
for i in asse_x:
    asse_y.append(2*A(i, x_0, gamma))
ax.plot(asse_x, asse_y, color = 'red', label='analitica')
ax.legend()

print(scarto(t, x, A0=x_0, omega=omega, gamma=gamma))

plt.show()
