import os
import math
import time
import numpy as np # модуль масивів
from scipy.interpolate import CubicSpline #кубічна інтерполяція
from scipy.optimize import minimize #знах екстремумів
import matplotlib as mpl # модуль відображення графіків
import matplotlib.pyplot as plt
from collections import deque  # імпортує клас черги collections.deque

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
    if (local_min[i] < 0) or (local_min[i] > 1000):
        local_min = np.delete(local_min,i)

    if(math.fabs(local_min[i]-local_min[i-1])<0.1):
        local_min = np.delete(local_min, i)

local_max.sort()
for i in range(local_max.size-1,0,-1):
    if (local_max[i] < 0) or (local_max[i] > 1000):
        local_max = np.delete(local_max,i)

    elif(math.fabs(local_max[i]-local_max[i-1]) < 0.1):
        local_max = np.delete(local_max, i)

print ("min")
print(local_min)
print("")
print("max")
print(local_max)
print("")

fig = plt.figure() #"фігура, що буде відображ.

meas_dots = plt.plot(x, measuration, 'o', label='data') #вхідні точки висот
min_dots = plt.plot(local_min, splines(local_min), 'g*', label='data') #екстремуми
max_dots = plt.plot(local_max, splines(local_max), 'r*', label='data') #екстремуми
spl_lines = plt.plot(dx, splines(dx)) #сплайни

plt.title('Затоплення')
plt.ylabel('Висоти')
plt.xlabel('Крок')

plt.grid(True)

save('pic_1_5_1', fmt='pdf')
save('pic_1_5_1', fmt='png')

plt.show()

water_lines = plt.plot([],[],'bo')
