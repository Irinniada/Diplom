import os
import math
import time
import numpy as np # модуль масивів
from scipy.interpolate import CubicSpline #кубічна інтерполяція
from scipy.optimize import minimize #знах екстремумів
import matplotlib as mpl # модуль відображення графіків
import matplotlib.pyplot as plt


# create new figure
fig = plt.figure()
grid = plt.GridSpec(10, 10)

# rain plot
max_precipitation = np.max(scaled_precipitation) if np.max(scaled_precipitation) > 0 else 0.01
ax_rain = plt.subplot(grid[:3, 0:])
ax_rain.set_ylabel('rain')
ax_rain.set_ylim(0, max_precipitation)
ax_rain.plot(np.arange(0, m), scaled_precipitation)
vertical_line, = ax_rain.plot([0, 0, 0, 0], lw=2, color='orange')
ax_rain.xaxis.set_visible(False)

# moisture plot
ax_moisture = plt.subplot(grid[4:, :-1])
ax_moisture.set_xlabel('moisture')
ax_moisture.set_ylabel('depth')
y_axis_limit = np.min(result) - 0.001 if np.min(result) != np.nan and np.min(result) != np.Inf else theta_r

# if we get nan values in solution - set limits manually
try:
    ax_moisture.set_xlim(y_axis_limit, theta_s)
except ValueError:
    ax_moisture.set_xlim(theta_r, theta_s)

ax_moisture.set_ylim(-l, 0)
moisture_line, = ax_moisture.plot([], [])
legend_text = ax_moisture.text(0.15, 0.95,
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=ax_moisture.transAxes,
                               s="t = " + str(0 * tau) + " (days)")

# integral bar
ax_integral = plt.subplot(grid[4:, -1:])
integral_bar, = ax_integral.bar(1, 0)
ax_integral.xaxis.set_visible(False)
ax_integral.yaxis.set_label_position("right")
ax_integral.yaxis.tick_right()
s_text = ax_integral.text(0.5, 0.5, s="S(0.5 m) = " + str(0), rotation='vertical',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=ax_integral.transAxes,)
# set bar limitations
s0 = []
for j in range(m):
    s0.append(h * np.sum(result[0:int(n / 2 + 1), j]))
ymin = np.min(s0) - 0.001 if np.min(s0) != np.nan and np.min(s0) != np.Inf else l * theta_r
ymax = np.max(s0) + 0.001 if np.max(s0) != np.nan and np.min(s0) != np.Inf else l * theta_s

# if we get nan values in solution - set limits manually
try:
    ax_integral.set_ylim(ymin, ymax)
except ValueError:
    ax_integral.set_ylim(l * theta_r, l * theta_s)

# values of x for moisture plot
x_values = np.linspace(0, -l, n)

# animation event
def animate(i):
    # get current moisture value
    x_moisture = result[:, i]

    # clear prev frame
    ax_moisture.fill_betweenx(x_values, theta_r, theta_s, facecolor='white')

    # draw a marker on rain graph
    vertical_line.set_data([i, i], [0, max_precipitation])

    # draw moisture graph
    moisture_line.set_data(x_moisture, x_values)
    legend_text.set_text("t = " + str(round(i * tau, self.output_precise)) + " (days)")

    # fill data
    moisture_fill = ax_moisture.fill_betweenx(x_values, theta_r, x_moisture, alpha=0.5, facecolor='C0')

    # draw integral value
    s = h * np.sum(x_moisture[0:int(n / 2 + 1)])
    s_text.set_text("S(0.5 m) = " + str(round(s, 5)))
    integral_bar.set_height(s)

    # return list of objects with new state
    return vertical_line, moisture_line, integral_bar, legend_text, s_text, moisture_fill,

# create, save and show animation
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, m), interval=200, blit=True, save_count=m + 1)
if save_result_to_gif:
    anim.save(soil_type + "-" + str(datetime.datetime.now()) + '.gif', writer="imagemagick",
              extra_args="convert", fps=5)
plt.show()
