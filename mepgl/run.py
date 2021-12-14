#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np

from config import (
    F,
    N,
    a_1,
    a_2,
    ax,
    ay,
    b_1,
    b_2,
    comp_domain,
    default_relaxation_step_number,
    delta,
    dx,
    eta,
    gamma,
    h,
    iterations,
    m_xx_1,
    m_xx_2,
    m_yy_1,
    m_yy_2,
    modes,
    multicomponent,
    q_1,
    q_2,
    sc_domain,
    simulation_name,
    u_1,
    u_2,
    v_1,
    v_2,
    x,
    y,
)
from mepgl_lib.launcher_utils import generate_init, launch_simulation, write_header

data_type = np.float32

# Parse input
parser = argparse.ArgumentParser(description="Initialization.")
parser.add_argument("-d", "--debug", help="debug build", action="store_true")
parser.add_argument("-n", "--noinit", help="No init", action="store_true")
args = parser.parse_args()

# Makes simulation directory and copy the config file
sim_data_dir = f"./simulations/{simulation_name}/"
# sim_output_dir = f"./output/{simulation_name}/"

os.makedirs(sim_data_dir, exist_ok=True)
# os.makedirs(sim_output_dir, exist_ok=True)

shutil.copy(f"./config.py", f"{sim_data_dir}/config.py")
# shutil.copy(f"./config.py", f"{sim_output_dir}/config.py")

try:
    shutil.copy(
        f"./batched_params.json", f"{sim_data_dir}/batched_params.json"
        # f"./batched_params.json", f"{sim_output_dir}/batched_params.json"
    )
except:
    pass


print(f"Simulation name: {simulation_name}")

# Generate initial guess
if not args.noinit:
    print("[*] Generating init files. ")
    print("Saving input data...     ", end=" ")
    generate_init(
        F,
        a_1,
        a_2,
        ax,
        ay,
        b_1,
        b_2,
        comp_domain,
        delta,
        eta,
        gamma,
        h,
        m_xx_1,
        m_xx_2,
        m_yy_1,
        m_yy_2,
        multicomponent,
        q_1,
        q_2,
        sc_domain,
        simulation_name,
        u_1,
        u_2,
        v_1,
        v_2,
        x,
        y,
    )
    print("done")
    print("===> Initialization finished <==")
    print("")

# Generate the header file
print("[*] Generating header.")
write_header(N, dx, default_relaxation_step_number, multicomponent)

# Compile binaries
print("[*] Compiling binaries.")
if args.debug:
    subprocess.run(["bash", "./compile.sh", "--debug"])
else:
    subprocess.run(["bash", "./compile.sh"])

print("===> Compilation finished <==")
print("")

print("[*] Running simulation. ")
launch_simulation(simulation_name, F, iterations, modes)

print("[*] Post processing. ")
subprocess.run(["python", "./post.py"])
