#in questo file applico RK alla risoluzione dell'equazione dell'oscillatore armonico smorzato e forzato nel caso sottosmorzato con forzante = Acos(omega_f*t)
#uso omega_f != omega per rendere la soluzione del problema più complicata

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def risolvi(t_fine, t_inizio, N, x_0, dx_0, omega, gamma, omega_f, A): #funzione per la risoluzione dell'equazione differenziale
    def sistema(t, v, omega, gamma, omega_f, A): #sistema di ODE da risolvere
        x, y = v

        def forzante(t, omega_f, A):
            return A*np.cos(omega_f*t)
        
        def f(t, x, y, omega, gamma, omega_f, A): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
            return - gamma*y - (omega**2)*x + forzante(t, omega_f, A)
        
        dx_dt = y #x' = g(t, x, y) = y
        dy_dt = f(t, x, y, omega, gamma, omega_f, A) #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, omega, gamma, omega_f, A): #matrice jacobiana del sistema
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
                    args = (omega, gamma, omega_f, A), #parametri di f
                    first_step = (t_fine-t_inizio)/1e5, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=1e-8, #errore relativo massimo accettabile
                    atol=1e-8*A/(omega**2), #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t.tolist(), x.tolist(), y.tolist()  

def soluzione(t, t0, x0, y0, omega, gamma, omega_f, A): #soluzione esatta dell'equazione differenziale
    #soluzione omogenea
    omega_d = np.sqrt(omega**2 - (gamma/2)**2)
    
    def omogenea(t, C, K): #caso sottosmorzato
        return np.exp(-gamma*t/2) * (C*np.cos(omega_d*t) + K*np.sin(omega_d*t))

    #soluzione particolare
    ampiezza = A / np.sqrt((omega**2 - omega_f**2)**2 + (gamma*omega_f)**2)
    fase = np.arctan2(gamma*omega_f, omega**2 - omega_f**2)
    
    def particolare(t):
        return ampiezza*np.cos(omega_f*t - fase)

    def d_particolare(t): #derivata prima rispetto al tempo della particolare
        return -ampiezza*omega_f*np.sin(omega_f*t - fase)

    #risolvo il sistema per trovare le costanti
    matrice = np.array([[np.cos(omega_d*t0), np.sin(omega_d*t0)],
                        [-(gamma/2)*np.cos(omega_d*t0) - omega_d*np.sin(omega_d*t0), -(gamma/2)*np.sin(omega_d*t0) + omega_d*np.cos(omega_d*t0)]])
    vettore = np.exp(gamma*t0/2) * np.array([x0-particolare(t0), y0-d_particolare(t0)])
    C, K = np.linalg.solve(matrice, vettore) #risolve un sistema del tipo matrice*(c, k) = vettore
    return omogenea(t, C, K) + particolare(t)

def scarto(t, x, **parametri): #scarto relativo tra la soluzione e l'approssimazione
    scarto=[]
    somma=0
    for i in range(len(t)):
        if i==0: #il primo valore di x è zero per la condizione iniziale
            scarto.append(0)
        else:
            sol = soluzione(t[i], **parametri)
            scarto.append(np.abs((sol-x[i])/(sol)))
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto relativo")
    ax.set_xlabel("Tempo (s)")
    ax.set_yscale("log")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t) #divido la somma per il numero di punti per normalizzare

#parametri per Runge Kutta
t_fine = 1e-4 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
N = 100000 #numero di punti in cui si vuole la soluzione
x_0 = 0 #x(t_inzio)
dx_0 = 0 #x'(t_inzio)

#parametri oscillatore e della forzante
omega = 1e7 #pulsazione propria dell'oscillatore (Hz)
gamma = 1 #coefficiente di smorzamento (Hz)
omega_f = 1.8e7 #pulsazione della forzante (Hz)
A = 1e-7 #ampiezza della forzante (l'ho dell'ordine di grandezza giusto per avere x della stessa dimensione di quello del caso con l'onda gravitazionale)

#grafico della soluzione
t, x, dx = risolvi(t_fine, t_inizio, N, x_0, dx_0, omega, gamma, omega_f, A)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione")
ax.set_xlabel("Tempo (s)")
ax.plot(t, x, color = 'blue', label="RK")
asse_x=np.linspace(t_fine, t_inizio, N)
asse_y=[]
for i in asse_x:
    asse_y.append(soluzione(i, t_inizio, x_0, dx_0, omega, gamma, omega_f, A))
ax.plot(asse_x, asse_y, color="red", label="analitica")
ax.legend()

print(scarto(t, x, t0=t_inizio, x0=x_0, y0=dx_0, omega=omega, gamma=gamma, omega_f=omega_f, A=A))

plt.show()
