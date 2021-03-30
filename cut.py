#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system
"""

import argparse
import os
import shutil

# Parse input
parser = argparse.ArgumentParser(description='Cut frames')
parser.add_argument('-s', '--simulation', help="simulation to process")
parser.add_argument('-i', '--interval', help="specify interval to cut")
args = parser.parse_args()


if (not args.simulation) or (not args.interval):
    exit(1)

i, f = args.interval.split("-")
i = int(i)
f = int(f)
simulation_name = args.simulation

print(f"Cut frames {i} to {f} in simulation {simulation_name}")

simdir = f"./simulations/{simulation_name}/"

if os.path.exists(simdir+"output_data_bak"):
    shutil.rmtree(simdir+"output_data_bak")
shutil.copytree(simdir+"output_data", simdir+"output_data_bak", )

d = []
for (dirpath, dirnames, filenames) in os.walk(simdir+"output_data"):
    d.extend(dirnames)
    break

d_int = [int(n) for n in d]

F = max(d_int) + 1

for n in range(i, f):
    shutil.rmtree(simdir+"output_data/"+f"{n}")

for n in range(f, F):
    shutil.move(simdir+"output_data/"+f"{n}", simdir+"output_data/"+f"{n-(f-i)}")
