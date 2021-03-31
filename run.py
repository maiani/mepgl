#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system
"""

import argparse
import subprocess

import numpy as np
import os
import shutil

from config import simulation_name

data_type = np.float32

# Parse input
parser = argparse.ArgumentParser(description='Initialization.')
parser.add_argument('-d', '--debug', help="debug build", action='store_true')
parser.add_argument('-n', '--noinit', help="No init", action='store_true')
args = parser.parse_args()

# Makes simulation directory and copy the config file
sim_dir_name = f"./simulations/{simulation_name}/"
os.makedirs(sim_dir_name, exist_ok=True)
shutil.copy(f"./config.py", f"./simulations/{simulation_name}/config.py")
try:
    shutil.copy(f"./batched_params.json", f"./simulations/{simulation_name}/batched_params.json")
except:
    pass

# Generate initial guess
if not args.noinit:
    print("[*] Generating init files ")
    import gen_init

# Generate the header file
print("[*] Generating header")
import gen_header

# Compile binaries
print("[*] Compiling binaries")
if args.debug:
    subprocess.run(["bash", "./compile.sh", "--debug"])
else:
    subprocess.run(["bash", "./compile.sh"])    

print("")
print("[*] Generating launcher")
import gen_launcher

print("===> Initialization finished <==")
print("")
print("[*] Running simulation")

# Run the launcher
subprocess.run(["bash", "./launcher.sh"])
