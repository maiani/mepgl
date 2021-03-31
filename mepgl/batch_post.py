#!/usr/bin/env python3
import json
import subprocess

import numpy as np

gamma = np.linspace(-0.5, 0.5, 11)

for i in np.ndindex(gamma.shape[0]):
    print(f"gamma = {gamma[i]:+4.2f}""")

for i in np.ndindex(gamma.shape[0]):
    batched_params_dict = dict()
    batched_params_dict['gamma'] = gamma[i]    

    batched_params_json = json.dumps(batched_params_dict)

    batched_params_file = open("./batched_params.json","w+")
    batched_params_file.write(batched_params_json)
    batched_params_file.close()

    #subprocess.call("./run.py")
    subprocess.run(["python", "./post.py" ])
