#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot([], [], ':+')

ax.set_title("MFEP")
ax.set_xlabel("s")
ax.set_ylabel("F")

def update(frame):
    e = np.loadtxt("energy.csv", delimiter=',')
    F = e.shape[0]
    x = np.linspace(0,1,F)
    line.set_data(x, e)
    deltaFn = np.max(e-e[0])
    deltaFe = np.max(e-e[-1])
    ax.set_title(rf"$\Delta F_n =${deltaFn:.6}     $\Delta F_e =${deltaFe:.6}")
    fig.gca().relim()
    fig.gca().autoscale_view()
    return line,

animation = FuncAnimation(fig, update, interval=1000)

plt.show()

