#in questo file simulo il segnale generato da un'onda gravitazionale ad ampiezza costante e frequenza pari a quella di risonanza con una durata infinita

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.signal import find_peaks
import equazione_differenziale as ode

class GW: #onda gravitazionale
    def __init__(self):
        #parametri relativi all'onda
        self.inizio = [-1, 100] #tempo di inizio del segnale (s)
        self.pulsazione = [2*np.pi*5e6] #pulsazione dell'onda (Hz)
        self.ampiezza = [1e-21] #ampiezza dell'onda (strain)
        self.fase = -np.pi/2 #fase (radianti)

    def h(self, t): #funzione che descrive l'andamento di h nel tempo
        for j in range(len(onda.inizio)-1):
            if t<onda.inizio[j+1] and t>=onda.inizio[j]: #capisco quale onda monocromatica usare
                indice = j
        return self.ampiezza[indice] * np.cos(self.pulsazione[indice]*t + self.fase)

    def dd_h(self, t): #derivata seconda della unzione che descrive l'andamento di h nel tempo
        for j in range(len(onda.inizio)-1):
            if t<onda.inizio[j+1] and t>=onda.inizio[j]: #capisco quale onda monocromatica usare
                indice = j
        return -(self.pulsazione[indice]**2) * self.ampiezza[indice] * np.cos(self.pulsazione[indice]*t + self.fase) 


class BAW: #classe che descrive le proprietà del cristallo e del modo di vibrazione scelto
    def __init__(self):
        #parametri relativi al cristallo
        self.d = 1e-3 #spessore del cristallo (m)
        
        #parametri relativi al modo di vibrazione
        self.n = 3 #n del modo di vibrazione (m=0, p=0, n dispari)
        self.omega = 2*np.pi*5e6 #pulsazione del modo di vibrazione (Hz)
        self.gamma = 2*np.pi*5e6/1e7 #coefficente di smorzamento del modo di vibrazione(Hz)
        self.eta_x = 0.1 #trapping parameter asse x (adimensionale)
        self.eta_y = 0.1 #trapping parameter asse y (adimensionale)
        self.k = 1e-2 #coefficiente di accoppiamento elettromeccanico (C/m)
            
    def xi(self): #parametro di accoppiamento onda-cavità (n dispari, m=p=0)
        costante = 16/((self.n**2)*(np.pi**2))
        numeratore = erf(np.sqrt(self.n)*self.eta_x)*erf(np.sqrt(self.n)*self.eta_y)
        denominatore = erf(np.sqrt(2*self.n)*self.eta_x)*erf(np.sqrt(2*self.n)*self.eta_y)
        return (self.d/2)*costante*numeratore/denominatore
    
    def corrente(self, dB): #funzione che restiruisce la corrente prodotta dal cristallo data la derivata rispetto al tempo della parte temporale displacement (dB)
        I = []
        for i in dB:
            I.append(self.k*i)
        return I


def ampiezza(t, x): #funzione che calcola l'ampiezza della soluzione approssimata
    massimi, _ = find_peaks(x)
    minimi, _ = find_peaks(-np.array(x))

    tempi = []
    ampiezze = []
    for i in range(min([len(massimi), len(minimi)])):
        tempi.append(np.mean([t[massimi[i]], t[minimi[i]]]))
        ampiezze.append(x[massimi[i]] - x[minimi[i]])
    return tempi, ampiezze

#valori iniziali
t_fine = 6 #tempo di fine (s)
t_inizio = 0 #tempo di inizio (s)
B_0 = 0 #B(0) (m)
dB_0 = 0 #B'(0) (m/s)

onda = GW()
cristallo = BAW()

#dato che in 6 s ci sono tantissime oscillazioni non posso simulare tutto il segnale quindi lo simulo a segmenti per ricavere l'ampiezza a tempi diversi
tempi_B = []
ampiezze_B = []
tempi_I = []
ampiezze_I = []
durata = 1e-6 #durata di ogni segmento
for inizio in np.linspace(t_inizio, t_fine-durata, 100): #tempi di inizio dei segmenti
    t, B, dB = ode.risolvi_analitica(inizio+durata, inizio, t_inizio, B_0, dB_0, 10000, onda, cristallo) #calcolo la parte temporale del displacement (B) e la sua derivata rispetto al tempo (dB)
    I = cristallo.corrente(dB) #calcolo la corrente prodotta dal cristallo
    tempo_B, amp_B = ampiezza(t, B)
    tempo_B.pop(0)
    tempo_B.pop(-1)
    amp_B.pop(0)
    amp_B.pop(-1)
    tempi_B = tempi_B + tempo_B
    ampiezze_B = ampiezze_B + amp_B
    tempo_I, amp_I = ampiezza(t, I)
    tempo_I.pop(0)
    tempo_I.pop(-1)
    amp_I.pop(0)
    amp_I.pop(-1)
    tempi_I = tempi_I + tempo_I
    ampiezze_I = ampiezze_I + amp_I


#grafici del segnale
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex=True)
ax[1].set_xlabel("Tempo [s]")

#grafico del displacement
ax[0].set_ylabel("Ampiezza del displacement [m]") #B ha l'unità di misura di una lunghezza invece U è adimensionale
ax[0].grid(True)
ax[0].plot(tempi_B, ampiezze_B, color = 'blue')

#grafico della corrente
ax[1].set_ylabel("Ampiezza della corrente [A]")
ax[1].grid(True)
ax[1].plot(tempi_I, ampiezze_I, color = 'blue')

#durata della crescita
regime_B = max(ampiezze_B) #ampiezze di regime
regime_I = max(ampiezze_I)
print(regime_B)
print(regime_I)
i=0
while ampiezze_B[i]<regime_B*0.99:
    i=i+1
print(tempi_B[i]) #tempo in cui si raggiunge il 99% del regime

plt.show()
