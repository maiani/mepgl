#!/usr/bin/env python3

"""
Reload the outputdata of a simulation as inputdata.
"""

import shutil

from config import simulation_name

print("Reloading simulation data...   ", end="")
shutil.rmtree(f"./simulations/{simulation_name}/input_data")
shutil.copytree(f"./simulations/{simulation_name}/output_data", f"./simulations/{simulation_name}/input_data")

print("done.")
