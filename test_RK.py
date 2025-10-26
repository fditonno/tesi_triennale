#questo file serve solo a provare che la funzione che implementa il metodo RK funzioni
'''
applico il metodo RK4 alla risoluzione della seguente ODE:
2x'' + 4x' + 10x = 5sin(2t)
'''

import matplotlib.pyplot as plt
import numpy as np
from equazioni_differenziali import Eulero_secondo_ordine as E
from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK

def f(t, x, y, a, b, c, A, w): #l'ODE deve essere nella forma x''=f
    return (A*np.sin(w*t)-b*y-c*x)/a

def soluzione(t): #soluzione esatta dell'equazione differenziale
    A = 10*np.cos(2*t)/17 + 5*np.sin(2*t)/34
    B = -10*np.cos(2*t)/17
    C = 5*np.sin(2*t)/34
    return np.exp(-t)*A+B+C

#condizioni iniziali
x0 = 0
y0 = 0

fig, ax = plt.subplots (nrows = 1, ncols = 1)

#grafico della soluzione esatta
t = np.linspace(6, 0, 1000)
x = []
for i in t:
    x.append(soluzione(i))
ax.plot(t, x, label="soluzione", color = 'blue')

#grafico con metodo di Eulero
t, x, y = E(6, 0, 0.1, x0, y0, f, a=2, b=4, c=10, A=5, w=2)
ax.errorbar(t, x, fmt='o', ls='none', label="Eulero", color = 'red')

#grafico della soluzione approssimata con RK4
t, x, y = RK(6, 0, 0.1, x0, y0, f, a=2, b=4, c=10, A=5, w=2)
ax.errorbar(t, x, fmt='o', ls='none', label="RK4", color = 'green')
#in questo caso ho usato un passo grande per poter distinguere i punti, diminuendo il passo si riduce l'errore

ax.legend()
plt.show()
