import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.integrate import solve_ivp

#costanti fisiche
c = 299792458 #velocità della luce (m/s)
G = 6.6743e-11 #costante di gravitazione universale (N*m^2/Kg^2) 

class GW: #onda gravitazionale
    def __init__(self):
        #parametri relativi all'onda
        self.media = 20e-6 #media della gaussiana (s)
        self.sigma = 5e-6 #sigma della gaussiana (s)
        self.A = 1e-25 #ordne di grandezza dell'ampiezza dell'onda
        self.fase = 0 #fase (radianti)

    def frequenza(self, t): #funzione che descrive la variazione della frequenza dell'onda nel tempo
        return 5e6

    def d_frequenza(self, t): #derivata prima della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return 0

    def dd_frequenza(self, t): #derivata seconda della funzione che descrive la variazione della frequenza dell'onda nel tempo
        return 0

    def ampiezza(self, t): #funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return (self.A/np.sqrt(2*np.pi*(self.sigma**2))) * np.exp(-(1/2)*((t-self.media)/self.sigma)**2)

    def d_ampiezza(self, t): #derivata prima della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return self.ampiezza(t) * (-(t-self.media)/(self.sigma**2))

    def dd_ampiezza(self, t): #derivata seconda della funzione che descrive la variazione dell'ampiezza dell'onda nel tempo
        return -self.ampiezza(t)/(self.sigma**2) - ((t-self.media)/(self.sigma**2)) * self.d_ampiezza(t)

    def h(self, t): #funzione che descrive l'andamento di h nel tempo
        return self.ampiezza(t)*np.cos(2*np.pi*self.frequenza(t)*t + self.fase)

    def dd_h(self, t): #derivata seconda della unzione che descrive l'andamento di h nel tempo
        A = self.ampiezza(t)
        d_A = self.d_ampiezza(t)
        dd_A = self.dd_ampiezza(t)
        omega = 2*np.pi*self.frequenza(t)
        d_omega = 2*np.pi*self.d_frequenza(t)
        dd_omega = 2*np.pi*self.dd_frequenza(t)
        argomento = omega*t + self.fase
        d_argomento = omega + d_omega*t
        dd_argomento =  2*d_omega + dd_omega*t
        return dd_A*np.cos(argomento) - 2*d_A*d_argomento*np.sin(argomento) - A*((d_argomento)**2)*np.cos(argomento) - A*dd_argomento*np.sin(argomento)

 
class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #parametri relativi al cristallo
        self.d = 1e-3 #spessore del cristallo (m)
        self.L = 14e-3 #raggio del cristallo (m)
        self.l = 7e-3 #raggio degli elettrodi (m)
        self.R = 600e-3 #raggio di curvatura della superficie del cristallo (m)
        
        #parametri relativi al modo di vibrazione scelto
        self.n = 3 #n del modo di vibrazione scelto (m=0, p=0, n dispari)
        self.omega = 2*np.pi*5e6 #pulsazione del modo di vibrazione scelto (Hz)
        self.gamma = 2*np.pi*5e6/1e7 #coefficente di smorzamento del modo scelto (Hz)
        self.eta_x = 1 #trapping parameter asse x (adimensionale)
        self.eta_y = 1 #trapping parameter asse y (adimensionale)
        self.k = 1e-2 #coefficiente di accoppiamento elettromeccanico (C/m)
            
    def xi(self): #parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*self.eta_x)*erf(np.sqrt(self.n)*self.eta_y)
        denominatore = erf(np.sqrt(2*self.n)*self.eta_x)*erf(np.sqrt(2*self.n)*self.eta_y)
        return (self.d/2)*costante*numeratore/denominatore

    def alfa(self):
        return (2/np.pi)*(self.eta_x/self.L)**2

    def beta(self):
        return (2/np.pi)*(self.eta_y/self.L)**2

    def Ux(self, x): #dipendenza da x della parte spaziale del displacement (m=p=0)
        return np.exp(-np.pi*self.n*self.alfa()*(x**2)/2)

    def Uy(self, y): #dipendenza da y della parte spaziale del displacement (m=p=0)
        return np.exp(-np.pi*self.n*self.beta()*(y**2)/2)
    
    def corrente(self, dB): #funzione che restiruisce la corrente prodotta dal cristallo data la derivata rispetto al tempo della parte temporale displacement (dB)
        integrale_x, _ = quad(self.Ux, -self.l, self.l) #integrale di Ux tra -l e l
        integrale_y, _ = quad(self.Uy, -self.l, self.l) #integrale di Uy tra -l e l
        #cost = self.k * integrale_x * integrale_y
        cost = self.k #credo che l'integrale sia già dentro k
        I = []
        for i in dB:
            I.append(cost*i)
        return I


def risolvi(t_fine, t_inizio, x_0, dx_0, N, rtol, atol, onda, cristallo): #funzione per la risoluzione dell'equazione differenziale        
    def sistema(t, v, onda, omega, gamma, xi): #sistema di ODE da risolvere
        x, y = v

        def f(t, x, y, onda, omega, gamma, xi): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
            '''
            omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
            xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
            '''
            return (onda.dd_h(t)*xi/2) - gamma*y - (omega**2)*x
        
        dx_dt = y #x' = g(t, x, y) = y
        dy_dt = f(t, x, y, onda, omega, gamma, xi) #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, onda, omega, gamma, xi): #matrice jacobiana del sistema
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
                    args = (onda, cristallo.omega, cristallo.gamma, cristallo.xi()), #parametri di f
                    first_step = (t_fine-t_inizio)/1e4, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=rtol, #errore relativo massimo accettabile
                    atol=atol, #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t, x, y

def scarto(t, B, dB, onda, cristallo): #scarto relativo tra la parte sinistra e destra dell'equazione differenziale
    xi = cristallo.xi()
    ddB = np.gradient(dB, t) #calcolo numericamente la derivata seconda d B

    def sinistra(B, dB, ddB): #la parte sinistra contiene B e le sue derivate quindi è quella che viene approssimata
        return ddB + cristallo.gamma*dB + B*(cristallo.omega)**2

    def destra(t): #la parte destra non presenta approssimazioni quindi può essere usata come valore vero
        return onda.dd_h(t)*xi/2

    scarto=[]
    somma=0
    for i in range(len(t)):
        des = destra(t[i])
        scarto.append(np.abs((sinistra(B[i], dB[i], ddB[i])-des)/des))
        somma = somma + scarto[i]

    #grafico dello scarto
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.set_title("Scarto relativo")
    ax.set_xlabel("Tempo (s)")
    ax.set_yscale("log")
    ax.plot(t, scarto, color = 'green')
    return somma/len(t) #divido la somma per il numero di punti per normalizzare

#valori iniziali
t_fine = 80e-6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
B_0 = 0 #B(t_inzio) (m)
dB_0 = 0 #B'(t_inzio) (m/s)

#parametri per Runge Kutta
N=100000 #numero di punti in cui si vuole la soluzione, non influisce sulla precisione di RK
rtol = 1e-8 #errore relativo massimo che si vuole sui punti della soluzione (1e-8 è il valore che sembra funzionare meglio)
atol = rtol*1e-23 #errore assoluto massimo che si vuole sui punti della soluzione (errore relativo * ordine di grandezza di B)

onda = GW()
cristallo = BAW()
t, B, dB = risolvi(t_fine, t_inizio, B_0, dB_0, N, rtol, atol, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
#usare dB calcolato da RK semra funzionare meglio rispetto ad usare B derivato con np.gradient quindi uso quello di RK
I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo

#grafico di h
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Onda (h)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Strain")
x = np.linspace(t_fine, t_inizio, N)
y = []
for i in x:
    y.append(onda.h(i))
ax.plot(x, y, color = 'blue')

#grafico della derivata seconda di h
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Derivata seconda dell'onda (ddh)")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Strain/s^2")
x = np.linspace(t_fine, t_inizio, N)
y = []
for i in x:
    y.append(onda.dd_h(i))
ax.plot(x, y, color = 'blue')

#grafico del displacement
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("B")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Displacement (m)") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax.plot(t, B, color = 'blue')

#grafico della corrente
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.set_title("Segnale")
ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Corrente (A)")
ax.plot(t, I, color = 'blue')

#print(scarto(t, B, dB, onda, cristallo)) #non sembra che funzioni tanto bene il metodo per calcolare lo scarto

plt.show()
