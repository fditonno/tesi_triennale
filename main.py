import numpy as np
from equazioni_differenziali import Runge_Kutta_secondo_ordine as RK

class GW: #onda gravitazionale che si propaga in direzione generica
    def __init__(self, theta, phi, a, b, omega): #basterà modificare il costruttore per aggiungere o togliere parametri necessari alle funzioni plus e cross
        self.theta = theta #angolo della direzione di propagazione rispetto all'asse x
        self.phi = phi #angolo della direzione di propagazione rispetto all'asse z
        self.a = a
        self.b = b
        self.omega = omega

    def plus(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di h+ nel tempo (messa a caso per ora)
        return self.a*np.sin(self.omega*t)

    def cross(self, t): #derivata seconda rispetto al tempo della funzione che descrive l'andamento di hx nel tempo (messa a caso per ora)
        return self.b*np.cos(self.omega*t)

    def h(self, t): #darivata seconda rispetto al tempo del tensore h
        p = self.plus(t)
        c = self.cross(t)

        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)

        h00 = p*(sin_phi*sin_theta + cos_phi)
        h01 = c*cos_phi
        h02 = c*sin_phi*sin_theta
        h11 = p*(sin_phi*cos_theta - cos_phi)
        h12 = c*sin_phi*cos_theta
        h22 = -p*(sin_phi*cos_theta + sin_phi*sin_theta)
        h = [ #è simmetrica
            [h00, h01, h02],
            [h01, h11, h12],
            [h02, h12, h22]]
        return h

def f(t, x, y, onda, i, j, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, x')
    '''
    i e j sono gli indici per i tensori
    omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
    xi è il tensore che dà l'accoppiamento tra l'onda e la cavità
    '''
    h = onda.h(t) #derivata seconda rispetto al tempo del tensore h, rappresenta la forza apparente esercitata sul cristallo dall'onda
    return (h[i][j]*xi[i][j])/2 - gamma*y - (omega**2)*x

def risolvi(t_fine, t0, N, x0, dev_x0, y0, dev_y0, onda, cristallo, X, n): #funzione che restituisce la soluzione dell'equazione differenziale per un certo modo di vibrazione
    '''
    t_fine e t0 sono gli estremi dell'intervallo in cui si vuole la soluzione
    N è il numero di punti in cui si vuole l'approssimazione della soluzione, più è grande più l'approssimazione è precisa
    x_0 è x(t=t0) e y(0) è x'(t=t0), dev_x0 e dev_y0 sono le deviazioni standard associate a questi valori
    X definisce il modo di vibrazione: X=0 modo A, X=1 modo B, X=2 modo C
    n è numero dell'armonica considerato
    '''
    passo = (t_fine-t0)/N
    j=2 #le componenti di xi sono diverse da 0 solo se j=2
    i=X #gli spostamenti dalle posizioni di equilibrio per il modo A sono dirette lungo x, per il modo B lungo y e per il modo C lungo z
    #calcolo omega, gamma e xi fuori da f per evitare che vengano ricalcolate ad ogni iterazione di RK
    omega = cristallo.omega(X, n) #devo ancora scrivere la classe cristallo
    gamma = cristallo.gamma(X, n)
    xi = cristallo.xi(X, n)
    return RK(t_fine, t0, passo, x0, dev_x0, y0, dev_y0, f, onda, i, j, omega, gamma, xi)


#parametri onda
phi = 0 #onda diretta lungo z
theta = 0
a = 2
b = 1
omega = 3

onda = GW(theta, phi, a, b, omega)
print(onda.h(1))
