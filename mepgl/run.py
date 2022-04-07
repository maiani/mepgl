#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np

from config import (F, N, a_1, a_2, ax, ay, b_1, b_2, comp_domain,
                    default_relaxation_step_number, delta, dev_number,
                    double_precision, dx, eta, gamma, h, iterations, m_xx_1,
                    m_xx_2, m_yy_1, m_yy_2, modes, multicomponent, q_1, q_2,
                    sc_domain, simulation_name, thin_film, u_1, u_2, v_1, v_2,
                    x, y)
from mepgl_lib.launcher_utils import (generate_headers, generate_init,
                                      launch_simulation)

data_type = np.float32

# Parse input
parser = argparse.ArgumentParser(description="Initialization.")
parser.add_argument("-d", "--debug", help="debug build", action="store_true")
parser.add_argument("-ni", "--no-init", help="no initialization", action="store_true", dest="no_init")
parser.add_argument("-rc", "--reload-continue", help="reload and continue", action="store_true", dest="rel_cont")
args = parser.parse_args()

sim_data_dir = f"./simulations/{simulation_name}/"
print(f"Simulation name: {simulation_name}")

# Generate initial guess
if not (args.no_init or args.rel_cont):

    # Makes simulation directory and copy the config file
    os.makedirs(sim_data_dir, exist_ok=True)
    shutil.copy(f"./config.py", f"{sim_data_dir}/config.py")
    try:
        shutil.copy(
            f"./batched_params.json",
            f"{sim_data_dir}/batched_params.json"
        )
    except:
        pass
        
    print("[*] Generating init files. ")
    print("Saving input data...    ", end="")
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
    print("done.")
    print("==> Initialization finished.")
    print("")

    # Generate the header file
    new_headers = generate_headers(
        N=N,
        dx=dx,
        default_relaxation_step_number=default_relaxation_step_number,
        multicomponent=multicomponent,
        thin_film=thin_film,
        double_precision=double_precision,
        dev_number=dev_number,
    )

    if new_headers or not os.path.exists("./mepgl"):
        # Compile binaries
        print("[*] New header generated or missing executable, compiling binaries.")
        if args.debug:
            subprocess.run(["bash", "./compile.sh", "--debug"])
        else:
            subprocess.run(["bash", "./compile.sh"])
        print("==> Compilation finished.")
        print("")
    else:
        print("[*] No new header needed, skipping compilation.")

if args.rel_cont:    
    print("Reloading simulation data...   ", end="")
    shutil.rmtree(f"./simulations/{simulation_name}/input_data")
    shutil.copytree(f"./simulations/{simulation_name}/output_data", f"./simulations/{simulation_name}/input_data")
    print("done.")
    print("")

print("[*] Running simulation.")
#generate_launcher(simulation_name, F, iterations, modes)
launch_simulation(simulation_name, F, iterations, modes)

print("[*] Post processing.")
subprocess.run(["python", "./post.py"])
