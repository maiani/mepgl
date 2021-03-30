#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system
"""

import os
import shutil

import numpy as np

from config import (simulation_name, F, x, y, comp_domain, sc_domain,
                    a_1, b_1, m_xx_1, m_yy_1, a_2, b_2, m_xx_2, m_yy_2, 
                    h, u_1, v_1, u_2, v_2, ax, ay, 
                    q_1, q_2, eta, gamma, delta,
                    multicomponent)

data_type = np.float64
#data_type = np.float32

sim_dir_name = f"./simulations/{simulation_name}/"
os.makedirs(sim_dir_name, exist_ok=True)

print("Saving input data...     ", end=" ")
input_dir = sim_dir_name + "input_data/"

if os.path.exists(input_dir):
    shutil.rmtree(input_dir)
os.makedirs(input_dir, exist_ok=True)

np.save(input_dir + f"x.npy", x.astype(data_type))
np.save(input_dir + f"y.npy", y.astype(data_type))
np.save(input_dir + f"comp_domain.npy", comp_domain.astype(np.int32))
np.save(input_dir + f"sc_domain.npy", sc_domain.astype(np.int32))
np.save(input_dir + f"a_1.npy", a_1.astype(data_type))
np.save(input_dir + f"b_1.npy", b_1.astype(data_type))
np.save(input_dir + f"m_xx_1.npy", m_xx_1.astype(data_type))
np.save(input_dir + f"m_yy_1.npy", m_yy_1.astype(data_type))
np.save(input_dir + f"a_2.npy", a_2.astype(data_type))
np.save(input_dir + f"b_2.npy", b_2.astype(data_type))
np.save(input_dir + f"m_xx_2.npy", m_xx_2.astype(data_type))
np.save(input_dir + f"m_yy_2.npy", m_yy_2.astype(data_type))
np.save(input_dir + f"h.npy", h.astype(data_type))

q = np.array([q_1, q_2, eta, gamma, delta])
np.save(input_dir + f"q.npy", q.astype(data_type))


for n in range(F[0]):
    os.makedirs(input_dir + f"{n}/", exist_ok=True)
    np.save(input_dir + f"{n}/ax.npy", ax[n].astype(data_type))
    np.save(input_dir + f"{n}/ay.npy", ay[n].astype(data_type))
    np.save(input_dir + f"{n}/u_1.npy", u_1[n].astype(data_type))
    np.save(input_dir + f"{n}/v_1.npy", v_1[n].astype(data_type))
    
    if multicomponent:
        np.save(input_dir + f"{n}/u_2.npy", u_2[n].astype(data_type))
        np.save(input_dir + f"{n}/v_2.npy", v_2[n].astype(data_type))
 
print("done")
