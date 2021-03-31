#!/usr/bin/env python3
import os
import shutil

import numpy as np

r_1   = np.array([4.0])
r_2   = np.array([5.0, 6.0, 7.0, 8.0])
h   = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])

os.makedirs("results", exist_ok=True)

for i,j,k in np.ndindex(r_1.shape[0], r_2.shape[0], h.shape[0]):
    simulation_name = f"r{r_1[i]:3.1f}-R{r_2[j]:3.1f}-h{h[k]:4.2f}-integer"

    sim_dir_name = f"./simulations/{simulation_name}/"
    try:
        shutil.copy(f"./simulations/{simulation_name}/{simulation_name}.npz", f"./results/{simulation_name}.npz")
    except:
        pass
