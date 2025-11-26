#in questo file applico RK alla risoluzione dell'equazione di Eulero Cauchy
#f è simile al caso dell'oscillatore e dipende esplicitamente dal tempo, ho usato parametri che si avvicinano ai valori omega e gamma

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def risolvi(t_fine, t_inizio, N, x_0, dx_0, alfa, beta): #funzione per la risoluzione dell'equazione differenziale
    def sistema(t, v, alfa, beta): #sistema di ODE da risolvere
        x, y = v
        
        def f(t, x, y, alfa, beta): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
            return - alfa*y/t - beta*x/(t**2)
        
        dx_dt = y #x' = g(t, x, y) = y
        dy_dt = f(t, x, y, alfa, beta) #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, alfa, beta): #matrice jacobiana del sistema
        '''
        [[dg/dx, dg/dy],
        [df/dx, df/dy]]
        '''
        return [[0, 1],
                [-beta/(t**2), -alfa/t]]

    sol = solve_ivp(sistema,
                    [t_inizio, t_fine],
                    [x_0, dx_0], #condizioni iniziali x_0 = x(t_inizio), dx_0 = x'(t_inizio)
                    method = 'Radau', #metodi: RK45, DOP853, Radau, BDF
                    t_eval = np.linspace(t_inizio, t_fine, N), #punti in cui si vuole la soluzione
                    args = (alfa, beta), #parametri di f
                    first_step = (t_fine-t_inizio)/1e4, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=1e-8, #errore relativo massimo accettabile
                    atol=1e-8*x_0, #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t.tolist(), x.tolist(), y.tolist()

def soluzione(t, t0, x0, y0, alfa, beta): #soluzione esatta dell'equazione differenziale
    radici = np.roots([1, alfa-1, beta]) #risolvo l'equazione caratteristica
    r1, r2 = radici[0], radici[1]

    if np.iscomplex(r1): #caso di soluzioni complesse (r1,r2 = a ± ib)
        a = r1.real
        b = abs(r1.imag)

        #funzioni che compaiono nella soluzione
        def f(t): return (t**a) * np.cos(b*np.log(t))
        def g(t): return (t**a) * np.sin(b*np.log(t))

        #derivate delle funzioni
        def df(t): return (t**(a-1)) * (a*np.cos(b*np.log(t)) - b*np.sin(b*np.log(t)))
        def dg(t): return (t**(a-1)) * (a*np.sin(b*np.log(t)) + b*np.cos(b*np.log(t)))

    elif np.isclose(r1, r2): #caso di soluzioni coincidenti (r1=r2 reali)
        def f(t): return (t**r)
        def g(t): return (t**r) * np.log(t)

        def df(t): return r*(t**(r-1))
        def dg(t): return (t**(r-1)) * (r*np.log(t) + 1)

    else: #caso di soluzioni reali distinte
        def f(t): return t**r1
        def g(t): return t**r2

        def df(t): return r1*(t**(r1 - 1))
        def dg(t): return r2*(t**(r2 - 1))

    #risolvo il sistema lineare per trovare i vaori delle costanti
    matrice = np.array([[f(t0), g(t0)],
                          [df(t0), dg(t0)]])
    vettore = np.array([x0, y0])
    c, k = np.linalg.solve(matrice, vettore) #risolve un sistema del tipo matrice*(c, k) = vettore
    return c*f(t) + k*g(t)

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
    ax.set_yscale("log")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t) #divido la somma per il numero di punti per normalizzare

#parametri per Runge Kutta
t_fine = 1e2 #per questo valore di t_fine il parametro che moltiplica y nell'equazione sarà 2 ordini di grandezza minore di gamma (1)
t_inizio = 1e-7 #per questo valore di t_inizio il parametro che moltiplica x nell'equazione sarà 2 ordini di grandezza maggiore di omega^2 (1e14)
N = 100000 #numero di punti in cui si vuole la soluzione
x_0 = 1e-21 #x(t_inzio) (l'ho dell'ordine di grandezza giusto per avere x della stessa dimensione di quello del caso con l'onda gravitazionale)
dx_0 = 0 #x'(t_inzio)

#parametri per l'equazione differenziale
beta = 100
alfa = 0.7

#grafico della soluzione
t, x, dx = risolvi(t_fine, t_inizio, N, x_0, dx_0, alfa, beta)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Soluzione")
ax.plot(t, x, color = 'blue', label='RK')
asse_x=np.linspace(t_fine, t_inizio, N)
asse_y=[]
for i in asse_x:
    asse_y.append(soluzione(i, t_inizio, x_0, dx_0, alfa, beta))
ax.plot(asse_x, asse_y, color="red", label='analitica')
ax.legend()

print(scarto(t, x, t0=t_inizio, x0=x_0, y0=dx_0, alfa=alfa, beta=beta))

plt.show()
