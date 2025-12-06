#questa libreria contiene le funzioni che risolvono l'equazione differenziale per il calcolo del displacement

import numpy as np
from scipy.integrate import solve_ivp

def risolvi_analitica(t_fine, t_inizio, t_0, x_0, dx_0, N, onda, cristallo):
    '''
    Funzione che restituisce la soluzione dell'equazione differenziale: B" + gamma*B' + (omega^2)*B = xi*ddh/2 nel caso in cui essa presenta soluzione analitica:
    si considera un'onda gravitazionale fatta da una successione di onde monocromatiche.
    t_fine e t_inizio danno l'intervallo temporale nel quale si vuole la soluzione e N è il numero di punti in cui si vuole la soluzione.
    x_0 è B(t_0) e dx_0 è B'(t_0).
    '''
    def soluzione(t, t0, x0, y0, omega, gamma, omega_f, ampiezza_f, fase_f): #soluzione esatta dell'equazione differenziale
        #soluzione omogenea
        omega_d = np.sqrt(omega**2 - (gamma/2)**2)
    
        def omogenea(t, C, K): #caso sottosmorzato
            return np.exp(-gamma*t/2) * (C*np.cos(omega_d*t) + K*np.sin(omega_d*t))

        def d_omogenea(t, C, K): #derivata prima rispetto al tempo dell'omogenea
            omega_d = np.sqrt(omega**2 - (gamma/2)**2)
            return -(gamma/2) * omogenea(t, C, K) + np.exp(-gamma*t/2) * (-C*omega_d*np.sin(omega_d*t) + K*omega_d*np.cos(omega_d*t))

        #soluzione particolare
        ampiezza = ampiezza_f / np.sqrt((omega**2 - omega_f**2)**2 + (gamma*omega_f)**2)
        fase = np.arctan2(gamma*omega_f, omega**2 - omega_f**2)
    
        def particolare(t):
            return ampiezza*np.cos(omega_f*t - fase + fase_f)

        def d_particolare(t): #derivata prima rispetto al tempo della particolare
            return -ampiezza*omega_f*np.sin(omega_f*t - fase + fase_f)

        #risolvo il sistema per trovare le costanti
        matrice = np.array([[np.cos(omega_d*t0), np.sin(omega_d*t0)],
                            [-(gamma/2)*np.cos(omega_d*t0) - omega_d*np.sin(omega_d*t0), -(gamma/2)*np.sin(omega_d*t0) + omega_d*np.cos(omega_d*t0)]])
        vettore = np.exp(gamma*t0/2) * np.array([x0-particolare(t0), y0-d_particolare(t0)])
        C, K = np.linalg.solve(matrice, vettore) #risolve un sistema del tipo matrice*(c, k) = vettore
        return omogenea(t, C, K) + particolare(t), d_omogenea(t, C, K) + d_particolare(t) #restituisce la soluzione e la sua derivata prima

    xi = cristallo.xi()
    t=np.linspace(t_inizio, t_fine, N)
    x=[]
    y=[]
    indice_precedente = 0
    for i in range(len(t)):
        for j in range(len(onda.inizio)-1):
            if t[i]<onda.inizio[j+1] and t[i]>=onda.inizio[j]: #capisco quale onda monocromatica usare
                indice = j
        if indice != indice_precedente and i!=0:
            t_0 = t[i-1]
            x_0 = x[i-1]
            dx_0 = y[i-1]
        A = (xi/2) * (-(onda.pulsazione[indice]**2)*onda.ampiezza[indice]) #calcolo l'ampiezza della forzante
        a, b = soluzione(t[i], t_0, x_0, dx_0, cristallo.omega, cristallo.gamma, onda.pulsazione[indice], A, onda.fase)
        indice_precedente = indice
        x.append(a)
        y.append(b)
    return t, x, y

def risolvi_RK(t_fine, t_inizio, x_0, dx_0, N, rtol, atol, onda, cristallo):
    '''
    Funzione che approssima la soluzone dell'equazione differenziale: B" + gamma*B' + (omega^2)*B = xi*ddh/2 nei casi in cui essa non presenta soluzione analitica.
    t_fine e t_inizio danno l'intervallo temporale nel quale si vuole la soluzione e N è il numero di punti in cui si vuole la soluzione.
    x_0 è B(t_inizio) e dx_0 è B'(t_inizio).
    '''
    def sistema(t, v, onda, omega, gamma, xi, t_inizio): #sistema di ODE da risolvere
        x, y = v

        def f(t, x, y, onda, omega, gamma, xi, t_inizio): #funzione che compare nell'equazione differenziale scritta nella forma x"=f(t, x, y) dove y=x'
            '''
            omega e gamma sono la pulsazione e la larghezza di banda del modo normale considerato
            xi è il termine di accoppiamento tra la cavità e l'onda gravitazionale
            '''
            return (onda.dd_h(t, t_inizio)*xi/2) - gamma*y - (omega**2)*x
        
        dx_dt = y #x' = g(t, x, y) = y
        dy_dt = f(t, x, y, onda, omega, gamma, xi, t_inizio) #y' = f(t, x, y)
        return [dx_dt, dy_dt]

    def jacobiano(t, v, onda, omega, gamma, xi, t_inizio): #matrice jacobiana del sistema
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
                    args = (onda, cristallo.omega, cristallo.gamma, cristallo.xi(), t_inizio), #parametri di f
                    first_step = (t_fine-t_inizio)/1e4, #passo iniziale
                    max_step = np.inf, #massima dimensione del passo consentita
                    rtol=rtol, #errore relativo massimo accettabile
                    atol=atol, #errore assoluto massimo accettabile
                    jac=jacobiano) #matrice jacobiana del sistema
    x, y = sol.y                                             
    return sol.t, x, y

def scarto(t, B, dB, onda, cristallo): #non funziona tanto bene, risulta essere circa un'ordine di grandezza maggiore rispetto allo scarto vero nei casi con soluzione analitica se si usa N=1000000
    '''
    Funzione che calcola lo scarto relativo tra la parte sinistra e la parte destra dell'equazione differenziale nel caso in cui si usa la soluzione approssimata
    '''
    xi = cristallo.xi()
    ddB = np.gradient(dB, t) #calcolo numericamente la derivata seconda d B

    def sinistra(B, dB, ddB): #la parte sinistra contiene B e le sue derivate quindi è quella che viene approssimata
        return ddB + cristallo.gamma*dB + B*(cristallo.omega)**2

    def destra(t, t_inizio): #la parte destra non presenta approssimazioni quindi può essere usata come valore vero
        return onda.dd_h(t, t_inizio)*xi/2

    scarto=[]
    somma=0
    for i in range(len(t)):
        des = destra(t[i], t[0])
        scarto.append(np.abs((sinistra(B[i], dB[i], ddB[i])-des)/des))
        somma = somma + scarto[i]

    return scarto, somma/len(t) #divido la somma per il numero di punti per normalizzare
