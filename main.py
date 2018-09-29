import os
import math
import time
import numpy as np # модуль масивів
from scipy.interpolate import CubicSpline #кубічна інтерполяція
from scipy.optimize import minimize #знах екстремумів
import matplotlib as mpl # модуль відображення графіків
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque  # імпортує клас черги collections.deque
#import animation

#збереження картинки

def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = './pictures/{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)
    #plt.close()


#задаємо вхідні дані

N = 10 #
measuration = np.random.rand(N) #масив вхідних даних (висот) (зараз - рандом, взагалі задається "=np.array..."
measuration *= 10
measuration -= 5 #це все до випадкової генерації
print(measuration)

x = np.arange(0,N,1) #масив по осі Х

splines = CubicSpline(x, measuration) #шматки сплайнів
def spl(x_abs):
    temp = splines(x_abs)
    return temp[0]

def reverse_spl(x_abs):
    temp = splines(x_abs)
    return (-1)*temp[0]

dx = np.arange(0, N-1, 0.1) #розбиваємо х для відображ. графіку/сплайнів


#знах екстремуми, поки що для сплайнів
local_min = np.zeros(N)
local_max = np.zeros(N)

for i in range(N-1,-1,-1):
    print(i)

    current_spl = splines(dx[i * 10:])

    if (i == 0):
        if(current_spl[0]<current_spl[1]): #якщо лівий край нижче - не враховуємо
            print(local_min)
            print(local_max)
            local_min = np.delete(local_min, i)
            local_max = np.delete(local_max, i)
            print("left delete!")
            print(local_min)
            print(local_max)

        elif (current_spl[0]>current_spl[1]):        #якщо лівий край вижче - врах як максимум
            print(local_min)
            print(local_max)
            local_max[i] = x[0]
            local_min = np.delete(local_min, i)
            print("left max!")
            print(local_min)
            print(local_max)

    if (i == N-1):
        current_spl = splines(dx[(i-1) * 10:])

        if(current_spl[N-1]<current_spl[N-2]): #якщо правий край нижче - не враховуємо
            local_min = np.delete(local_min,i)
            local_max = np.delete(local_max, i)
            print("right delete!")

        elif (current_spl[N-1]>current_spl[N-2]):        #якщо правий край вижче - врах як максимум
            local_max[i] = x[N-1]
            local_min = np.delete(local_min, i)
            print("right max!")

    if ((i>0)&(i<N-1)):
        temp_min = minimize(spl, dx[i * 10:], method='nelder-mead')
        local_min[i] = temp_min.x[0]
        print ("min")
        print(local_min)
        print("")
        temp_max = minimize(reverse_spl, dx[i * 10:], method='nelder-mead')
        local_max[i] = temp_max.x[0]
        print("max")
        print(local_max)
        print("")


#відловлюємо значення екстр, які майже збігаються
print("Sorting...\n")
local_min.sort()
for i in range(local_min.size-1,0,-1):
    if (local_min[i] < 0):
        local_min = np.delete(local_min,i)
    if (local_min[i] > 1000):
        local_min = np.delete(local_min, i)
for i in range(local_min.size - 1, 0, -1):
    if(math.fabs(local_min[i]-local_min[i-1])<0.1):
        local_min = np.delete(local_min, i)

local_max.sort()
for i in range(local_max.size-1,0,-1):
    if (local_max[i] < 0):
        local_max = np.delete(local_max,i)
    if (local_max[i] > 1000):
        local_max = np.delete(local_max, i)
for i in range(local_max.size - 1, 0, -1):
    if(math.fabs(local_max[i]-local_max[i-1]) < 0.1):
        local_max = np.delete(local_max, i)

print ("min")
print(local_min)
print("")
print("max")
print(local_max)
print("")

#початкова функція лінії води
def water_startlevel(dx):
    y = np.zeros(dx.size);
    for x_t in range(dx.size):
        for i in range(local_min.size-1):
            if ((dx[x_t] > local_min[i]) & (dx[x_t] < local_min[i+1])):
                y[x_t]=((dx[x_t]-local_min[i])*(splines(local_min[i+1])-splines(local_min[i]))/(local_min[i+1]-local_min[i]))+splines(local_min[i])

    return y

delta_h = 0.01*(max(local_max)-min(local_min))
eps = 0.05
h = splines(local_min)
print("h")
print(h)
#y = water_startlevel(dx)
x_lr = local_min
x_lr = np.repeat(x_lr, 2)

def water_level(dx):
    global delta_h, eps, h, x_lr
    count = 0
    set_end = True
    while (set_end):  # обраховуємо висоту
        for i in range(local_min.size):
            V = (local_max[i+1] - local_max[i])
            delta_V = 0
            delta_xl = x_lr[i * 2]  # ліва точка
            delta_xr = x_lr[(i * 2) + 1]  # права точка
            old_h = h[i]
            print("hi")
            print(i)
            while (math.fabs(V - delta_V) > eps):
                h[i] = h[i] + delta_h

                while (math.fabs(splines(delta_xl) - h[i]) > eps):  # ліва точка
                    delta_xl = delta_xl - eps

                while (math.fabs(splines(delta_xr) - h[i]) > eps):  # ліва точка
                    delta_xr = delta_xr + eps

                delta_V = 0.5 * (h[i] - old_h) * (x_lr[(i * 2) + 1] - x_lr[i * 2] + delta_xr - delta_xl)
                count = count + 1
                print("pook")
                print(V)
                print(delta_V)
                if delta_V > V:
                    h[i] = h[i] - delta_h
                    delta_xl = delta_xl + eps
                    delta_xr = delta_xr - eps
                    delta_h = delta_h * 0.5
                    print("пук")

            x_lr[i * 2] = delta_xl          #оновлюємо х
            x_lr[(i * 2) + 1] = delta_xr    #для висоти води
            print("done")
        set_end = False

    #малюємо лінію
    y = np.zeros(dx.size);
    for x_t in range(dx.size):
        for i in range(x_lr.size-1):
            if ((dx[x_t] > local_min[i]) & (dx[x_t] < local_min[i+1])):
                y[x_t]=((dx[x_t]-local_min[i])*(splines(local_min[i+1])-splines(local_min[i]))/(local_min[i+1]-local_min[i]))+splines(local_min[i])
    print(y)
    #return y


while True:
    water_level(dx)



'''fig, ax = plt.subplots()
water_line, = ax.plot(dx, water_startlevel(dx)) #рівень води
meas_dots, = ax.plot(x, measuration, 'o', label='data') #вхідні точки висот
min_dots, = ax.plot(local_min, splines(local_min), 'g*', label='data') #екстремуми
max_dots, = ax.plot(local_max, splines(local_max), 'r*', label='data') #екстремуми
spl_lines, = ax.plot(dx, splines(dx)) #сплайни'''

'''def init():  # only required for blitting to give a clean slate.
    water_line.set_data([], [])
    return water_line,


def animate(i):
    water_line.set_ydata(water_level(dx))  # update the data.
    return water_line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1, blit=True, save_count=50)

#plt.show()
#fig = plt.figure() #"фігура, що буде відображ.



plt.title('Затоплення')
plt.ylabel('Висоти')
plt.xlabel('Крок')

#plt.grid(True)

#save('pic_1_5_1', fmt='pdf')
#save('pic_1_5_1', fmt='png')

plt.show()

#water_lines = plt.plot([],[],'bo')'''

