import os
import math
import time
import json
from random import random
import numpy as np  # модуль масивів
from scipy.interpolate import CubicSpline  # кубічна інтерполяція
from scipy.optimize import minimize  # знах екстремумів
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Filtration1

'''class WaterLevel(object):
    def __init__(self,k_filter):
        self.x_l = []
        self.y_l = []

    def build(self,k):
        self.x_l, self.y_l = water_level(k)'''



def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = './pictures/{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)
    # plt.close()

def modeling(X,Y,i):
    # wt = WaterLevel
    fig, (ax, av) = plt.subplots(2, sharex = True)
    ax.set_xlim(-1,10)
    water_line, = ax.plot(x_lr, water_startlevel(x_lr), 'b')  # рівень води
    meas_dots, = ax.plot(x, measuration, 'o', label='data')  # вхідні точки висот
    min_dots, = ax.plot(local_min, splines(local_min), 'g*', label='data')  # екстремуми
    max_dots, = ax.plot(local_max, splines(local_max), 'r*', label='data')  # екстремуми
    spl_lines, = ax.plot(dx, splines(dx))  # сплайни

    # s_text = ax.text(-4, 0.1, s="H = " + str(0))
    print("Draw?")

    # print('check',check_done)

    # print("XYR", x_result, y_result)

    def init():  # only required for blitting to give a clean slate.
        y_under_surface = -12  # splines(local_min[np.argmin(min_min)] - 10 * delta_h)
        water_line.set_data([], [])
        water_linefill = ax.fill_between(dx, splines(dx) - 1, y_under_surface, color='w')

        return water_line, water_linefill,

    def animate(i):
        print('t', i)
        #x_l, y_l = find_level(k, i)
        print("Gorodraw")
        y_under_surface = -12  # splines(local_min[np.argmin(min_min)] - 10 * delta_h)
        water_linefill = ax.fill_between(dx, splines(dx), y_under_surface, color='w')


        # water_line.set_data(x_result[i], y_result[i])  # update the data.
        water_line.set_data(X[i], Y[i])  # update the data.

        # print("XYR_i", x_result[i], y_result[i])
        # s_text.set_text("H = " + h_sub)
        # water_line.set_data(x_result[i], y_result[i])  # update the data.
        # print(x_result[i], y_result[i])
        # spl_lines, = ax.plot(dx, splines(dx))  # сплайни
        return water_line, water_linefill, meas_dots, min_dots, max_dots, spl_lines,
    t_local = i
    print(i)
    anim = animation.FuncAnimation(fig, animate, frames=i, init_func=init, interval=200, blit=True, save_count=30,repeat=False)

    V_temp = np.asarray(V_f_all)
    V_temp = V_temp.T

    for i in range(local_min.size):  # к-ть заглибин, к-ть графіків
         print('(np.arange(0, t, 1)).size')
         print((np.arange(0, (np.asarray(V_temp[i])).size,1)).size)
         print('V_temp[t]')
         print((np.asarray(V_temp[i])))
         av.plot(np.arange(0, (np.asarray(V_temp[i])).size, 1), V_temp[i], label="i="+str(i))
    av.legend(loc="upper right")
    # fig_filter.show()

    plt.title('Затоплення')
    plt.ylabel('Висоти')
    plt.xlabel('Крок')
    # V_temp = np.asarray(V_f_all)
    # V_temp = V_temp.T

    plt.show()


    #fig = plt.figure()
    #ax = plt.axes()

    #return V_f_all

def visualize(set_intense, k):
    global local_min, local_max, set_filled, delta_h, x_lr, h, V, eps, eta, x_result, y_result, check_done, V_f_all
    check_anim = True
    for i in range(local_min.size):
        min_min[i] = splines(local_min[i])
    for i in range(local_max.size):
        max_max[i] = splines(local_max[i])

    y_under_surface = splines(local_min[np.argmin(min_min)] - 10 * delta_h)

    for i in range(N - 1, -1, -1):
        current_spl = splines(dx[i * 10:])

        if not i:
            if current_spl[0] < current_spl[1]:  # якщо лівий край нижче - не враховуємо
                local_min = np.delete(local_min, i)
                local_max = np.delete(local_max, i)

            elif current_spl[0] > current_spl[1]:  # якщо лівий край вижче - врах як максимум
                local_max[i] = x[0]
                local_min = np.delete(local_min, i)

        if i == N - 1:
            current_spl = splines(dx[(i - 1) * 10:])

            if current_spl[N - 1] < current_spl[N - 2]:  # якщо правий край нижче - не враховуємо
                local_min = np.delete(local_min, i)
                local_max = np.delete(local_max, i)

            elif current_spl[N - 1] > current_spl[N - 2]:  # якщо правий край вижче - врах як максимум
                local_max[i] = x[N - 1]
                local_min = np.delete(local_min, i)

        if (i > 0) and (i < N - 1):
            temp_min = minimize(lambda spl: splines(spl)[0], dx[i * 10:], method='nelder-mead')
            local_min[i] = temp_min.x[0]
            temp_max = minimize(lambda rspl: (-1) * splines(rspl)[0], dx[i * 10:], method='nelder-mead')
            local_max[i] = temp_max.x[0]

    # відловлюємо значення екстр, які майже збігаються
    # print("Sorting...\n")
    local_min.sort()
    for i in range(local_min.size - 1, -1, -1):
        if local_min[i] < 0.e+0:
            local_min = np.delete(local_min, i)

        if local_min[i] > 20:
            local_min = np.delete(local_min, i)

    for i in range(local_min.size - 1, -1, -1):
        if math.fabs(local_min[i] - local_min[i - 1]) < 0.1:
            local_min = np.delete(local_min, i)

    local_max.sort()
    for i in range(local_max.size - 1, -1, -1):
        if local_max[i] < 0.e+0:
            local_max = np.delete(local_max, i)

        if local_max[i] > 20:
            local_max = np.delete(local_max, i)

    for i in range(local_max.size - 1, -1, -1):
        if math.fabs(local_max[i] - local_max[i - 1]) < 0.1:
            local_max = np.delete(local_max, i)

    set_filled = np.zeros(local_min.size, dtype=bool)  # показник заповненості ділянки
    #time.sleep(1)
    delta_h = 0.01 * (local_max[np.argmax(max_max)] - local_min[np.argmin(min_min)])

    h = splines(local_min)
    x_lr = np.array(np.repeat(local_min, 2))
    V = np.zeros(local_min.size)

    for i in range(V.size):
        V[i] = set_intense * 0.005 * (local_max[i + 1] - local_max[i])
    x_R, y_R, t = water_level(k)
    print("Got level!")
    modeling(x_R,y_R,t)



def water_startlevel(x_lr):
    y = np.zeros(x_lr.size)
    for i in range(x_lr.size):
        y[i] = splines(x_lr[i])
    return y


def water_level(k_filter):
    t = 0
    print('Start floodind')
    global delta_h, eps, eta, set_filled, local_min, local_max, x_lr, h, V, V_f_all, check_done
    R = []
    x_result = []
    y_result = []
    local_min_saved = local_min #для обрахунку фільтрації
    min_min = np.zeros(local_min.size)
    max_max = np.zeros(local_max.size)
    for i in range(local_min.size):
        min_min[i] = splines(local_min[i])
    for i in range(local_max.size):
        max_max[i] = splines(local_max[i])

    y_under_surface = splines(local_min[np.argmin(min_min)] - 20 * delta_h)

    over_max_l = False
    over_max_r = False
    i = 0

    l_contact = 0
    set_end = True

    while set_end:
        i = 0
        V_f_all.append([])
        while i < local_min.size:
            delta_V = 0
            delta_h = 0.001  # 0.1*(max(local_max)-min(local_min))
            delta_xl = x_lr[i * 2]  # ліва точка

            delta_xr = x_lr[(i * 2) + 1]  # права точка
            old_h = h[i]
            # площа контакту води та грунту
            # впливає на об'єм затоплення (Об.зат = Об.опадів-Об.фільтр-Об.випар., Об.фільтр~пл конт.
            x_count = 0
            for temp in range(dx.size):
                if dx[temp] > delta_xl:
                    x_count = temp
                    break
            # від лівого краю до дх
            l_contact = math.sqrt(
                math.pow(dx[x_count] - delta_xl, 2) + math.pow(splines(dx[x_count]) - splines(delta_xl), 2))
            x_step = x_count

            while dx[x_step] < delta_xr:
                x_step = x_step + 1
                if x_step >= dx.size:
                    break
                l_contact = (l_contact + math.sqrt(math.pow(dx[x_step] - dx[x_step - 1], 2) + math.pow(
                    splines(dx[x_step] - splines(dx[x_step - 1])), 2)))
            # l_contact= l_contact * 0.01

            # задача фільтр
            # H - напори
            x_step = x_count
            v_sum_filter = 0
            if math.fabs(splines(delta_xl) - splines(local_min[i])) > 0:  # якщо є заповнення
                #H = Filtration1.RS(math.fabs(splines(local_min[i]) - y_under_surface),
                #                   math.fabs((splines(delta_xl) - splines(local_min[i]))))
                H = np.zeros(2)
                H[0] = math.fabs((splines(delta_xl) - splines(local_min[i])))
                H[-1] = 1
                v_filter = k_filter * math.fabs((H[0] - H[-1]) / (splines(local_min[i]) - y_under_surface))
                v_sum_filter = v_filter

            # TODO чисельний вимір висоти затоплення на графіку
            # TODO вивести графіки для кожної ділянки зі зміною відфільтрр об'єму (по часу), у звіті - відсотки
            # знаходиомо об'єм фільтрований
            V_filter = (l_contact * v_sum_filter) * 0.001

            if set_filled[i]:  # область вже заповнена
                i = i + 1
            else:
                # V = 10 * (local_max[i + 1] - local_max[i])
                while math.fabs(V[i] - delta_V) > eps:

                    h[i] = h[i] + delta_h

                    while math.fabs(splines(delta_xl) - h[i]) > eps:  # ліва точка
                        delta_xl = delta_xl - eta

                        if delta_xl < local_max[i]:  # якщо перелив
                            over_max_l = True
                            break

                        if splines(delta_xl) > h[i]:
                            delta_xl = delta_xl + eta
                            eta = eta * 0.5

                    while math.fabs(splines(delta_xr) - h[i]) > eps:  # права точка
                        delta_xr = delta_xr + eta

                        if delta_xr > local_max[i + 1]:  # якщо перелив
                            over_max_r = True
                            break

                        if splines(delta_xr) > h[i]:
                            delta_xl = delta_xl - eta
                            eta = eta * 0.5

                    delta_V = 0.01 * (
                                (h[i] - old_h) * (x_lr[(i * 2) + 1] - x_lr[i * 2] + delta_xr - delta_xl) - V_filter)

                    if delta_V > V[i]:
                        h[i] = h[i] - 2 * delta_h
                        delta_xl = delta_xl + eta
                        delta_xr = delta_xr - eta
                        delta_h = delta_h * 0.5

                    # TODO замінити реалізацію заповнення: показник заповнення set_filled, для адекватного малювання

                    if over_max_l:  # якщо перелив зліва
                        # input('Press <l> to continue')
                        set_filled[i] = True
                        over_max_l = False

                        # TODO: зупинка в точці, перевірити переливи
                        if i == 0:  # якщо перелив за ліву точку, яка є краєм
                            if (local_min.size==1):
                                set_filled[i] = True
                                set_end = False
                            break

                        elif not set_filled[i - 1]:  # лівий край нижчий за наступну т лок макс
                            V[i - 1] = V[i - 1] + V[i]
                            V[i] = 0

                        elif set_filled[i - 1]:  # те саме, але  i - 1 заповнена

                            '''local_max = np.delete(local_max, i)
                            if (splines(local_min[i]) > splines(local_min[i - 1])):
                                local_min = np.delete(local_min, i)
                            else:
                                local_min = np.delete(local_min, i - 1)
                            x_lr = np.delete(x_lr, 2 * i - 1)
                            x_lr = np.delete(x_lr, 2 * i - 1)
                            # set_filled = np.delete(set_filled, i - 1)
                            set_filled[i - 1] = False
                            i = i - 1'''
                            if(splines(local_max[i-1])<splines(local_max[i])):
                                set_filled[i] = True
                                V[i - 1] = V[i - 1] + V[i]
                                V[i] = 0
                                print("Underwater left, i-1 - down")
                            else:
                                print("Underwater left, i-1 - up")

                                local_max = np.delete(local_max, i)
                                if (splines(local_min[i])>splines(local_min[i-1])):
                                    local_min = np.delete(local_min, i)
                                else:
                                    local_min = np.delete(local_min, i-1)
                                x_lr = np.delete(x_lr, 2 * i - 1)
                                x_lr = np.delete(x_lr, 2 * i - 1)
                                #set_filled = np.delete(set_filled, i - 1)
                                set_filled[i - 1] = False
                                i = i - 1
                                # time.sleep(5)
                        break

                    if over_max_r:  # якщо перелив справа
                        set_filled[i] = True
                        over_max_r = False

                        if i == (set_filled.size - 1):  # якщо перелив за праву точку, яка є краєм
                            if (local_min.size==1):
                                set_filled[i] = True
                            break

                        elif not set_filled[i + 1]:  # справа не залито
                            V[i + 1] = V[i + 1] + V[i]
                            V[i] = 0

                        elif set_filled[i + 1]:  # те саме, але  i + 1 заповнена
                            #TODO перевірити, чи локалмакс_і+1 вище і+2, якщо так - брейк, якщо ні - об'єднання
                            if (splines(local_max[i+1]) > splines(local_max[i+2])):
                                set_filled[i] = True
                                V[i + 1] = V[i + 1] + V[i]
                                V[i] = 0
                            else:
                                local_max = np.delete(local_max, i + 1)
                                if (splines(local_min[i])>splines(local_min[i+1])):
                                    local_min = np.delete(local_min, i)
                                else:
                                    local_min = np.delete(local_min, i+1)
                                x_lr = np.delete(x_lr, 2 * i + 1)
                                x_lr = np.delete(x_lr, 2 * i + 1)
                                #set_filled = np.delete(set_filled, i + 1)
                                set_filled[i] = False  # це було set_filled[i + 1]
                                # time.sleep(5)
                                # як варіант - взяти за дно точку максимуму між ними''''''
                        break

                x_lr[i * 2] = delta_xl  # оновлюємо х
                x_lr[(i * 2) + 1] = delta_xr  # для висоти води
                i = i + 1

        #print('set_filled', set_filled)

        set_end = False
        for p in range(set_filled.size):
            if not set_filled[p]:
                set_end = True

        # малюємо лінію
        y = np.zeros(x_lr.size)
        for i in range(x_lr.size):
            y[i] = splines(x_lr[i])

        print("x_lr", x_lr)
        print("y", y)
        print("filled:",set_filled)

        '''for p in range(local_min.size):
            h_sub = h_sub + " " + (y[2*p]-splines(local_min[p]))
            print("h_sub ", h_sub)'''

        for i in range(local_min.size):
            l_contact = 0

            #H0 = y[i] - splines(local_min[i]) #Filtration1.RS(math.fabs(splines(local_min[i]) - y_under_surface),
                #               math.fabs((splines(delta_xl) - splines(local_min[i]))))

            H0 = math.fabs((h[i] - splines(local_min[i])))
            print('H',H0)
            v_filter = k_filter * math.fabs((H0 - 1) / (splines(local_min[i]) - y_under_surface))
            print('v_filt',v_filter)
            if (v_filter < 0):
                v_filter = 0
            x_count = 0
            for temp in range(dx.size):
                if dx[temp] > x_lr[2*i]:
                    x_count = temp
                    break
            # від лівого краю до дх
            l_contact = math.sqrt(
                math.pow(dx[x_count] - x_lr[2*i], 2) + math.pow(splines(dx[x_count]) - splines(x_lr[2*i]), 2))
            x_step = x_count

            while dx[x_step] < x_lr[2*i]:
                x_step = x_step + 1
                if x_step >= dx.size:
                    break
                l_contact = (l_contact + math.sqrt(math.pow(dx[x_step] - dx[x_step - 1], 2) + math.pow(
                    splines(dx[x_step] - splines(dx[x_step - 1])), 2)))
            #print('l_contact', l_contact)
            V_f_all[t].append(v_filter*l_contact)
            #print('V_f_all_i', V_f_all)
        #print('V_f_all',V_f_all)

        #with open("res.txt", "a") as file:
        #    file.write(json.dumps({"x_lr": list(x_lr), "y": list(y)}) + "\n")
        x_result.append(x_lr)
        y_result.append(y)
        t = t + 1
        print('t', t)
    return x_result,y_result, t



RES_FILE = "./res.txt"
if os.path.exists(RES_FILE):  # видаляємо неактуальні дані (попередній запуск програми)
    os.remove(RES_FILE)

N = 10
measuration = [random() * 20 - 10 for i in range(N)]
# measuration = [0, 2, 4, 2, 0, 1, 3, 5, 3, 0.1]

x = np.arange(0, N, 1)  # масив по осі Х
splines = CubicSpline(x, measuration)  # шматки сплайнів
dx = np.arange(0, N - 1, 0.1)  # розбиваємо х для відображ. графіку/сплайнів

# знах екстремуми, поки що для сплайнів
local_min = np.zeros(N)
local_max = np.zeros(N)
max_max = np.zeros(local_max.size)
min_min = np.zeros(local_min.size)

delta_h = 0.01 * (local_max[np.argmax(max_max)] - local_min[np.argmin(min_min)])
eps = 0.005
eta = 0.01
h = splines(local_min)
x_lr = np.array(np.repeat(local_min, 2))
V = np.zeros(local_min.size)

y_under_surface = local_min[np.argmin(min_min)] - 10 * delta_h
# TODO інтенсивність set_intense з інтерфейсу
k_filter = 1    # TODO коефіцієнт фільтрації з інтерфейсу
V_f_all = []
local_min_reserve = []
local_max_reserve = []
check_done = True

