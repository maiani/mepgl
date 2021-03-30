#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system
"""

import shutil

from config import simulation_name

print("Reloading simulation data...   ", end="")
shutil.rmtree(f"./simulations/{simulation_name}/input_data")
shutil.copytree(f"./simulations/{simulation_name}/output_data", f"./simulations/{simulation_name}/input_data")

print("done.")
