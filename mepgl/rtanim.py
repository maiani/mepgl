#!/usr/bin/env python3

"""
Real time plotter for mepgl.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("mepgl.mplstyle")

# Create axis
fig, ax = plt.subplots(figsize=(4, 3), dpi = 200)

(line,) = ax.plot([], [], ":+")
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$F(s)$")

# Create empty variable
e = np.array([])

def update_frame(frame):
    # Try to read new data
    try:
        e_new = np.loadtxt("energy.csv", delimiter=",")
    except:
        e_new = None
    
    # Save data if succesfully loaded
    if e_new is not None:
        e = e_new
    
    # Update plot
    F = e.shape[0]
    x = np.linspace(0, 1, F)
    line.set_data(x, e)
    deltaFn = np.max(e - e[0])
    deltaFe = np.max(e - e[-1])
    ax.set_title(rf"$\Delta F_n =${deltaFn:.6}     $\Delta F_e =${deltaFe:.6}")
    fig.gca().relim()
    fig.gca().autoscale_view()
    return (line,)


animation = FuncAnimation(fig, update_frame, interval=1000)

plt.show()

