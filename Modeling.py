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


def save(name='', fmt='png'):
    pwd = os.getcwd()
    iPath = './pictures/{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)
    # plt.close()


def modeling(set_intense, k):
    global local_min, local_max, set_filled, delta_h, x_lr, h, V, eps, eta

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

        if local_min[i] > 1000:
            local_min = np.delete(local_min, i)

    for i in range(local_min.size - 1, -1, -1):
        if math.fabs(local_min[i] - local_min[i - 1]) < 0.1:
            local_min = np.delete(local_min, i)

    local_max.sort()
    for i in range(local_max.size - 1, -1, -1):
        if local_max[i] < 0.e+0:
            local_max = np.delete(local_max, i)

        if local_max[i] > 1000:
            local_max = np.delete(local_max, i)

    for i in range(local_max.size - 1, -1, -1):
        if math.fabs(local_max[i] - local_max[i - 1]) < 0.1:
            local_max = np.delete(local_max, i)

    set_filled = np.zeros(local_min.size, dtype=bool)  # показник заповненості ділянки
    time.sleep(1)
    delta_h = 0.01 * (max(local_max) - min(local_min))

    h = splines(local_min)
    x_lr = np.array(np.repeat(local_min, 2))
    V = np.zeros(local_min.size)

    for i in range(V.size):
        V[i] = set_intense * 0.005 * (local_max[i + 1] - local_max[i])

    fig, ax = plt.subplots()
    water_line, = ax.plot(x_lr, water_startlevel(x_lr), 'b')  # рівень води
    meas_dots, = ax.plot(x, measuration, 'o', label='data')  # вхідні точки висот
    min_dots, = ax.plot(local_min, splines(local_min), 'g*', label='data')  # екстремуми
    max_dots, = ax.plot(local_max, splines(local_max), 'r*', label='data')  # екстремуми
    spl_lines, = ax.plot(dx, splines(dx))  # сплайни

    def init():  # only required for blitting to give a clean slate.
        water_line.set_data([], [])
        return water_line,

    def animate(i):
        water_line.set_data(water_level(i, k))  # update the data.
        return water_line,

    ani = animation.FuncAnimation(fig, animate, frames=10, init_func=init, interval=200, blit=True, save_count=50)
    plt.title('Затоплення')
    plt.ylabel('Висоти')
    plt.xlabel('Крок')
    plt.show()


def water_startlevel(x_lr):
    y = np.zeros(x_lr.size)
    for i in range(x_lr.size):
        y[i] = splines(x_lr[i])
    return y


def water_level(step, k_filter):
    print(k_filter)
    global delta_h, eps, eta, set_filled, local_min, local_max, x_lr, h, V
    count = 0
    over_max_l = False
    over_max_r = False
    set_end = True
    while set_end:  # обраховуємо висоту
        # замінити for на while
        i = 0
        while i < local_min.size:
            if set_filled[i]:  # область вже заповнена
                i = i + 1
            else:
                # V = 10 * (local_max[i + 1] - local_max[i])
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
                    l_contact = (l_contact + math.sqrt(math.pow(dx[x_step] - dx[x_step - 1], 2) + math.pow(
                        splines(dx[x_step] - splines(dx[x_step - 1])), 2)))
                    x_step = x_step + 1

                # задача фільтр
                # H - напори
                x_step = x_count
                v_sum_filter = 0
                if math.fabs(splines(delta_xl) - splines(local_min[i])) > 0:  # якщо є заповнення
                    while dx[x_step] < delta_xr:
                        H = Filtration1.RS(math.fabs(splines(dx[x_step]) - y_under_surface),
                                           math.fabs((splines(delta_xl) - splines(local_min[i]))))
                        v_filter = k_filter * math.fabs((H[0] - H[-1]) / (splines(dx[x_step]) - y_under_surface))
                        v_sum_filter = v_sum_filter + v_filter
                        x_step = x_step + 1
                # TODO чисельний вимір висоти затоплення на графіку
                # TODO вивести графіки для кожної ділянки зі зміною відфільтрр об'єму (по часу), у звіті - відсотки
                # знаходиомо об'єм фільтрований
                V_filter = (l_contact * v_sum_filter) * 0.001

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
                    count = count + 1

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
                            continue

                        elif not set_filled[i - 1]:  # лівий край нижчий за наступну т лок макс
                            V[i - 1] = V[i - 1] + V[i]
                            V[i] = 0

                        elif set_filled[i - 1]:  # те саме, але  i - 1 заповнена
                            local_max = np.delete(local_max, i)
                            local_min = np.delete(local_min, i - 1)
                            x_lr = np.delete(x_lr, 2 * i - 1)
                            x_lr = np.delete(x_lr, 2 * i - 1)
                            set_filled = np.delete(set_filled, i - 1)
                            set_filled[i - 1] = False
                            i = i - 1
                            # time.sleep(5)
                        break

                    if over_max_r:  # якщо перелив справа
                        set_filled[i] = True
                        over_max_r = False

                        if i == (set_filled.size - 1):  # якщо перелив за праву точку, яка є краєм
                            break

                        elif not set_filled[i + 1]:  # справа не залито
                            V[i + 1] = V[i + 1] + V[i]

                        elif set_filled[i + 1]:  # те саме, але  i + 1 заповнена
                            local_max = np.delete(local_max, i + 1)
                            local_min = np.delete(local_min, i + 1)
                            x_lr = np.delete(x_lr, 2 * i + 2)
                            x_lr = np.delete(x_lr, 2 * i + 1)
                            set_filled = np.delete(set_filled, i + 1)
                            set_filled[i] = False  # це було set_filled[i + 1]
                            # time.sleep(5)
                            # як варіант - взяти за дно точку максимуму між ними
                        break

                x_lr[i * 2] = delta_xl  # оновлюємо х
                x_lr[(i * 2) + 1] = delta_xr  # для висоти води
                i = i + 1
        set_end = False
        time.sleep(1)

    # TODO: прибрати бокові лінії
    # малюємо лінію
    y = np.zeros(x_lr.size)
    for i in range(x_lr.size):
        y[i] = splines(x_lr[i])

    print("x_lr", x_lr)
    print("y", y)

    with open("res.txt", "a") as file:
        file.write(json.dumps({"x_lr":  list(x_lr), "y": list(y)}) + "\n")

    return x_lr, y


RES_FILE = "./res.txt"
if os.path.exists(RES_FILE):  # видаляємо неактуальні дані (попередній запуск програми)
    os.remove(RES_FILE)

N = 10
measuration = [random() * 20 - 10 for i in range(N)]
x = np.arange(0, N, 1)  # масив по осі Х
splines = CubicSpline(x, measuration)  # шматки сплайнів
dx = np.arange(0, N - 1, 0.1)  # розбиваємо х для відображ. графіку/сплайнів

# знах екстремуми, поки що для сплайнів
local_min = np.zeros(N)
local_max = np.zeros(N)

delta_h = 0.01 * (max(local_max) - min(local_min))
eps = 0.005
eta = 0.01
h = splines(local_min)
x_lr = np.array(np.repeat(local_min, 2))
V = np.zeros(local_min.size)
y_under_surface = min(local_min) - 10 * delta_h
# TODO інтенсивність set_intense з інтерфейсу
k_filter = 1    # TODO коефіцієнт фільтрації з інтерфейсу
