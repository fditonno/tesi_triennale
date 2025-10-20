#Libreria che implementa i metodi di risuluzione numerica delle equazioni differenziali

import numpy as np

def media_pesata(valori, pesi):
    #calcolo della media
    s_pesi = 0
    s = 0
    for i in range(len(valori)):
        s_pesi = s_pesi + pesi[i]
        s = s + valori[i]*pesi[i]
    mu = s/s_pesi

    #calcolo l'errore sulla media
    scarto = 0
    for i in range(len(valori)):
        scarto = scarto + pesi[i]*(valori[i]-mu)**2
    N=len(valori)
    bessel = (N-1)/N
    dev_mu = np.sqrt(scarto/(bessel*s_pesi))/np.sqrt(N)
    return  mu, dev_mu

def Eulero_primo_ordine(t_fine, t0, h, x0, f, **parametri):
    '''
    Funzione che utilizza il metodo di Eulero per approssimare la soluizione di un'equazione differenziale del tipo:
    x' = f(t, x, **parametri) con x(t0)=x0
    t_fine e t0 sono gli estremi dell'intervallo in cui si vuole approssimare la soluzione e h è il passo.
    '''
    t=np.arange(t0, t_fine+h, h)
    x=[x0]
    for i in range(1, len(t), 1):
        k = f(t[i-1], x[i-1], **parametri) #calcolo il coefficente angolare della retta tangente a x (che è f per definizone) nel punto (t[i-1], x[i-1])
        x.append(x[i-1] + h*k) #approssimo il valore assunto dalla funzione x al tempo t[i] con il valore assunto dalla retta tangente al tempo t[i]
    return t, x

def Runge_Kutta_primo_ordine(t_fine, t0, h, x0, dev_x0, f, **parametri):
    '''
    Funzione che utilizza il metodo Runge Kutta di quarto ordine (RK4) per approssimare la soluizione di un'equazione differenziale del tipo:
    x' = f(t, x, **parametri) con x(t0)=x0 con errore associato dev_x0
    t_fine e t0 sono gli estremi dell'intervallo in cui si vuole approssimare la soluzione e h è il passo.
    '''
    t=np.arange(t0, t_fine+h, h)
    x=[x0]
    dev_x=[dev_x0]
    for i in range(1, len(t), 1):
        #il metodo è analogo a quello di Eulero ma si calcolano 4 coefficenti angolari in punti divesi (compresi tra t[i-1] e t[i])
        k1 = f(t[i-1], x[i-1], **parametri)
        k2 = f(t[i-1]+h/2, x[i-1]+h*k1/2, **parametri)
        k3 = f(t[i-1]+h/2, x[i-1]+h*k2/2, **parametri)
        k4 = f(t[i], x[i-1]+h*k3, **parametri)
        
        k, dev_k = media_pesata([k1, k2, k3, k4], [1, 2, 2, 1]) #il coefficente angolare utilizzato è un'opportuna media pesata dei coefficenti calcolati in precedenza
        '''
        per stimare l'errore associato all'approssimazione ho pensato di utilizzare l'errore sulla media dei valori k1,k2,k3,k4 che restituisce l'errore sul valore k;
        considero solo l'errore dovuto allo scarto dei valori dalla media,
        non considero l'errore di x[i-1] (che si propagherebbe sui valori k1,k2,k3,k4) in questo passaggio
        '''
        x.append(x[i-1] + h*k)
        dev_x.append(np.sqrt(dev_x[i-1]**2 + h*dev_k**2)) #propago gli errori di x[i-1] e k sul valore di x[i]
    return t, x, dev_x

def Eulero_secondo_ordine(t_fine, t0, h, x0, y0, f, **parametri):
    '''
    Funzione che utilizza il metodo di Eulero per approssimare la soluizione di un'equazione differenziale del tipo:
    x'' = f(t, x, x', **parametri) con x(t0)=x0 e x'(t0)=y0
    t_fine e t0 sono gli estremi dell'intervallo in cui si vuole approssimare la soluzione e h è il passo.
    
    L'equazione può essere separata in un sistema di equazioni del primo ordine:
    x' = g(t, x, x')=y (1)
    y' = f(t, x, x', **parametri) (2)
    il metodo di Eulero si appliza approssimando i valori di x e di y passo-passo come fatto in precedenza ma utilizzando g per il calcolo di x e f per il calcolo di y.
    '''
    t=np.arange(t0, t_fine+h, h)
    x=[x0]
    y=[y0]
    for i in range(1, len(t), 1):
        #approssimo l'equazione 1
        l = y[i-1] #g(t, x, x') è banalmente uguale al valore di y
        x.append(x[i-1] + h*l)

        #approssimo l'equazione 2
        k = f(t[i-1], x[i-1], y[i-1], **parametri)
        y.append(y[i-1] + h*k)
    return t, x, y

def Runge_Kutta_secondo_ordine(t_fine, t0, h, x0, dev_x0, y0, dev_y0, f, **parametri):
    '''
    Funzione che utilizza il metodo Runge Kutta di quarto ordine (RK4) per approssimare la soluizione di un'equazione differenziale del tipo:
    x'' = f(t, x, x', **parametri) con x(t0)=x0 e x'(t0)=y0 con errori associati dev_x0 e dev_y0
    t_fine e t0 sono gli estremi dell'intervallo in cui si vuole approssimare la soluzione e h è il passo.
    
    L'equazione può essere separata in un sistema di equazioni del primo ordine:
    x' = g(t, x, x')=y (1)
    y' = f(t, x, x', **parametri) (2)
    il metodo Runge Kutta si appliza approssimando i valori di x e di y passo-passo come fatto in precedenza ma utilizzando g per il calcolo di x e f per il calcolo di y.
    '''
    t=np.arange(t0, t_fine+h, h)
    x=[x0]
    y=[y0]
    dev_x=[dev_x0]
    dev_y=[dev_y0]
    for i in range(1, len(t), 1):
        l1 = y[i-1]
        k1 = f(t[i-1], x[i-1], y[i-1], **parametri)

        l2 = y[i-1] + h*k1/2
        k2 = f(t[i-1] + h/2, x[i-1] + h*l1/2, y[i-1] + h*k1/2, **parametri)

        l3 = y[i-1] + h*k2/2
        k3 = f(t[i-1] + h/2, x[i-1] + h*l2/2, y[i-1] + h*k2/2, **parametri)

        l4 = y[i-1] + h*k3/2
        k4 = f(t[i], x[i-1] + h*l3, y[i-1] + h*k3, **parametri)

        l, dev_l = media_pesata([l1, l2, l3, l4], [1, 2, 2, 1])
        x.append(x[i-1] + h*l)
        dev_x.append(np.sqrt(dev_x[i-1]**2 + h*dev_l**2))
        
        k, dev_k = media_pesata([k1, k2, k3, k4], [1, 2, 2, 1])
        y.append(y[i-1] + h*k)
        dev_y.append(np.sqrt(dev_y[i-1]**2 + h*dev_k**2))
    return t, x, dev_x, y, dev_y
        
    
