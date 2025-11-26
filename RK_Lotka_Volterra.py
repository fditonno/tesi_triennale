#in questo file applico RK alla risoluzione dell'equazione di Lotka Volterra
#ho usato valori di alfa e beta paragonabili ad omega e gamma per far si che il problema sia stiff

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def risolvi(t_fine, t_inizio, N, x_0, y_0, alfa, beta, gamma, delta): #funzione per la risoluzione dell'equazione differenziale
    def sistema(t, v, alfa, beta, gamma, delta): #sistema di ODE da risolvere
        x, y = v
        dx_dt = alfa*x - beta*x*y #x' = g(t, x, y)
        dy_dt = -gamma*y + delta*x*y #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, alfa, beta, gamma, delta): #matrice jacobiana del sistema
        '''
        [[dg/dx, dg/dy],
        [df/dx, df/dy]]
        '''
        x, y = v
        return [[alfa - beta*y, beta*x],
                [delta*y, -gamma + delta*x]]

    sol = solve_ivp(sistema,
                    [t_inizio, t_fine],
                    [x_0, y_0], #condizioni iniziali x_0 = x(t_inizio), y_0 = y(t_inizio)
                    method = 'Radau', #metodi: RK45, DOP853, Radau, BDF
                    t_eval = np.linspace(t_inizio, t_fine, N), #punti in cui si vuole la soluzione
                    args = (alfa, beta, gamma, delta), #parametri del sistema
                    first_step = (t_fine-t_inizio)/1e4, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=1e-8, #errore relativo massimo accettabile
                    atol=1e-8*min([x_0, y_0]), #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t.tolist(), x.tolist(), y.tolist()

def scarto(t, x, y, x0, y0, alfa, beta, gamma, delta): #scarto relativo tra la soluzione e l'approssimazione
    def H(x, y, alfa, beta, gamma, delta): #costante del moto
        return -gamma*np.log(x) + delta*x - alfa*np.log(y) + beta*y

    analitica = H(x0, y0, alfa, beta, gamma, delta) #H dovrà rimanere sempre lo stesso lungo il moto, H(x0, y0) è il valore vero di H
    scarto=[]
    somma=0
    for i in range(len(t)):
        scarto.append(np.abs((analitica-H(x[i], y[i], alfa, beta, gamma, delta))/analitica)) #scarto tra H vero e H nel punto
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto relativo")
    ax.set_xlabel("Tempo (s)")
    ax.set_yscale("log")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t) #divido la somma per il numero di punti per normalizzare

#parametri per Runge Kutta
t_fine = 500e-7 #(s)
t_inizio = 0 #(s)
N = 100000 #numero di punti in cui si vuole la soluzione
x_0 = 100 #numero iniziale di prede
y_0 = 30 #numero iniziale di predatori

#parametri per l'equazione differenziale
alfa = 1e7 #crescita delle prede (l'ho messo dell'ordine di omega)
beta = 1 #cattura delle prede (l'ho messo dell'ordine di gamma)
gamma = 3e7 #morte dei predatori
delta = 1e6 #crescita dei predatori quando catturano una preda

#grafico della soluzione
t, x, y = risolvi(t_fine, t_inizio, N, x_0, y_0, alfa, beta, gamma, delta)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione (RK)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Numero prede", color = 'blue')
ax.plot(t, x, color = 'blue', label='prede')
ax2 = ax.twinx()
ax2.set_ylabel("Numero predatori", color = 'orange')
ax2.plot(t, y, color = 'orange', label='predatori')

print(scarto(t, x, y, x_0, y_0, alfa, beta, gamma, delta))

plt.show()
