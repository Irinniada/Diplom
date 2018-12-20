import run_off
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#y_n - нижня точка выдрызка; висота выдрызка
# M - к-ть ітерацій по часу


def RS(x_max, h_gr):
    fig, ax = plt.subplots()
    delta_x = 0.1
    y_0 = 0
    x = np.arange(0, x_max, delta_x)
    N = x.size
    M = 10
    tau = 1
    k = 1
    mu = 1

    a_2 = k / mu
    sigma = a_2 * tau / (delta_x * delta_x)
    a = np.zeros(N)
    for i in range(N):
        a[i] = sigma
    b = np.zeros(N)
    for i in range(N):
        b[i] = sigma
    c = np.zeros(N)
    for i in range(N):
        c[i] = 1 + 2 * sigma

    #початкові умови
    h = np.zeros(N)
    #гр умови
    h[0] = h_gr
    h[-1] = 1
    h = run_off.TDMA(a,b,c,h)
    '''h_plot1, = ax.plot(x, h, 'b')  # рівень води
    h_temp = np.asarray(h)
    print("")
    print("T=0")

    
    h_temp = np.around(h_temp, 3)
    print(h_temp.reshape(-1,10))'''



    #for i in range(M):
    h[0] = h_gr
    h = run_off.TDMA(a, b, c, h)
    '''#plt.plot(x,h,color = (0.1+i*0.05,0.2,0.3))
    #print("")
    #print("T=",i)

    h_temp = np.asarray(h)
    h_temp = np.around(h_temp, 3)
    #print(h_temp.reshape(-1, 10))'''

    return h
    #plt.show()

#виводить з обмеженням на знаки після коми
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"



