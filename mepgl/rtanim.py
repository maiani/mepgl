#!/usr/bin/env python3

"""
Real time plotter for mepgl.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("mepgl.mplstyle")

history_N = 30
fn_history = np.zeros(history_N)

# Create axis
fig, ax = plt.subplots(figsize=(4, 3), dpi = 200)

line, = ax.plot([], [], "-+")
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$F(s)$")

axin = ax.inset_axes([0.80, 0.80, 0.20, 0.20])
linein, = axin.plot(np.arange(history_N), fn_history, "-")

# Create empty variable
e = np.array([])
deltaFn = 0
deltaFe = 0

def update_frame(frame):
    global e, fn_history

    # Try to read new data
    try:
        e_new = np.loadtxt("energy.csv", delimiter=",")
    except:
        e_new = None
    
    # Save data if succesfully loaded
    if e_new is not None:
        e = e_new
        deltaFn = np.max(e - e[0])
        deltaFe = np.max(e - e[-1])
        fn_history = np.roll(fn_history, -1)
        fn_history[-1] = deltaFn
    
    # Update plot
    x = np.linspace(0, 1, e.shape[0])
    line.set_data(x, e)
    ax.set_title(rf"$\Delta F_n =${deltaFn:.6}     $\Delta F_e =${deltaFe:.6}")
    
    linein.set_data(np.arange(history_N), fn_history)
    axin.set_ylim(np.min(fn_history)*0.95, np.max(fn_history)*1.05)

    fig.gca().relim()
    fig.gca().autoscale_view()
    return (line,)


animation = FuncAnimation(fig, update_frame, interval=1000)

plt.show()

