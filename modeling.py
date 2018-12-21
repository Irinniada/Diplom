import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Water
def modeling(intens,k):

    '''t = 0
    check_done = True
    while check_done:
        x_result, y_result = water_level(t,k_filter)
        t = t + 1
        print('t',t)

    print('x_result',x_result)
    print('y_result', y_result)'''
    Water.visualize(intens,k)

    # wt = WaterLevel
    fig, ax = plt.subplots()
    water_line, = ax.plot(Water.x_lr, Water.water_startlevel(Water.x_lr), 'b')  # рівень води
    meas_dots, = ax.plot(Water.x, Water.measuration, 'o', label='data')  # вхідні точки висот
    min_dots, = ax.plot(Water.local_min, Water.splines(Water.local_min), 'g*', label='data')  # екстремуми
    max_dots, = ax.plot(Water.local_max, Water.splines(Water.local_max), 'r*', label='data')  # екстремуми
    spl_lines, = ax.plot(Water.dx, Water.splines(Water.dx))  # сплайни
    # s_text = ax.text(-4, 0.1, s="H = " + str(0))
    print("Draw?")

    # print('check',check_done)

    # print("XYR", x_result, y_result)

    def init():  # only required for blitting to give a clean slate.
        y_under_surface = -12  # splines(local_min[np.argmin(min_min)] - 10 * delta_h)
        water_line.set_data([], [])
        water_linefill = ax.fill_between(Water.dx, Water.splines(Water.dx) - 1, y_under_surface, color='w')

        return water_line, water_linefill,

    def animate(i):
        print('t', Water.t)
        #x_l, y_l = find_level(k, i)
        print("Gorodraw")
        y_under_surface = -12  # splines(local_min[np.argmin(min_min)] - 10 * delta_h)
        water_linefill = ax.fill_between(Water.dx, Water.splines(Water.dx), y_under_surface, color='w')


        # water_line.set_data(x_result[i], y_result[i])  # update the data.
        water_line.set_data([0,10], [0,i])  # update the data.

        # print("XYR_i", x_result[i], y_result[i])
        # s_text.set_text("H = " + h_sub)
        # water_line.set_data(x_result[i], y_result[i])  # update the data.
        # print(x_result[i], y_result[i])
        # spl_lines, = ax.plot(dx, splines(dx))  # сплайни
        return water_line, water_linefill, meas_dots, min_dots, max_dots, spl_lines,
    t_local = Water.t
    print(Water.t)
    anim = animation.FuncAnimation(fig, animate, frames=20, init_func=init, interval=50, blit=True, save_count=30,repeat=False)

    plt.title('Затоплення')
    plt.ylabel('Висоти')
    plt.xlabel('Крок')
    # V_temp = np.asarray(V_f_all)
    # V_temp = V_temp.T

    plt.show()