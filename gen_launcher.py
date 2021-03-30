#!/usr/bin/env python3

"""
Initializing code for string methods applied to 2D GL system
"""

from config import simulation_name, F, iterations, modes

def generate_launcher(simulation_name, F, iterations, modes):    

    P = F.shape[0]

    bashfile = "#!/bin/bash\n"

    for i in range(P-1):
        command_string = f"./mep-gl --simname {simulation_name} --mode {modes[i]} --Fin {F[i]} --Fout {F[i+1]} --iterations {iterations[i]}\n ./reload.py \n"
        bashfile += command_string
        
    command_string = f"./mep-gl --simname {simulation_name} --mode {modes[-1]} --Fin {F[-1]} --Fout {F[-1]}  --iterations {iterations[-1]}\n"
    bashfile += command_string
    
    launcher_file = open("./launcher.sh","w+")
    launcher_file.write(bashfile)
    launcher_file.close()    
    
print("Writing launcher file....", end="")
generate_launcher(simulation_name, F, iterations, modes)
print("done.")