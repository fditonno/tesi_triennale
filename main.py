import numpy as np

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


#parametri onda
phi = 0 #onda diretta lungo z
theta = 0
a = 2
b = 1
omega = 3

onda = GW(theta, phi, a, b, omega)
print(onda.h(1))
