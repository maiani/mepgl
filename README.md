# MEPGL

Gauged string method for the computation of minimum energy paths (MEP) in Ginzburg-Landau (GL) models for superconductors and superfluids.

## Getting Started

### Prerequisites:
To run this software, you need

* CUDA Toolkit
* ZLib
* Boost libraries
* Python scientific libraries (numpy, scipy, matplotlib)

A conda yml file is provided to get these library, except for the CUDA toolkit.
To use it, run 
``` 
conda env create -f environment.yml
conda activate mepgl
```

You need to activate the `mepgl` environment each session, before running the code.

### Setting precision
To set the numeric precision, you need to edit `./src/real.cuh` and `./mepgl_lib/launcher_utils.py`

### Writing down a config file
The first step to run the program is to edit the configuration file `./config.py`. 
This file provide the parameters of the system and the information to build the initial guess for the path.

### Run the program

Once the config file is set up, just run the simulation with `./run.py`.
If you need to run it in debug mode: `./run.py --debug` 

Runtime commands (press the key and then Enter):
- `+` and `-` change the number of relaxation steps.
- `q` stops the simulation and save the results.
- `C` switch on and off NLCG.

The results of the simulation can be found in the `./simulations` folder under a directory with the name of the simulation.
 
### Details of the launcher 
TBW

### Run simulations in batches
TBW

### Other options
- If you want to continue a simulation run `./reload.py` to load the output of the previous run as initial guess, 
then run the simulation with `./run.py --noinit` to prevent overwriting the input files.

- You can check the mep in realtime while the simulation is running using `./rtanim.py`. 

## References
This code has been used for the following papers:

* \[1\] Benfenati A., Maiani A., Rybakov F. N., Bababev E. - *Vortex nucleation barrier revisited* - https://arxiv.org/abs/1911.09513
* \[2\] Maiani A., Benfenati A., Bababev E. - *Vortex nucleation barriers and surface fractional vortices in two-component Ginzburg-Landau model* 

If you use this code, please cite as 

```
@article{mepgl,
author = {Benfenati, Andrea and Maiani, Andrea and Rybakov, Filipp N. and Babaev, Egor},
doi = {10.1103/PhysRevB.101.220505},
issn = {24699969},
journal = {Phys. Rev. B},
month = {jun},
number = {22},
pages = {220505},
title = {{Vortex nucleation barrier in superconductors beyond the Bean-Livingston approximation: A numerical approach for the sphaleron problem in a gauge theory}},
url = {http://arxiv.org/abs/1911.09513 https://link.aps.org/doi/10.1103/PhysRevB.101.220505},
volume = {101},
year = {2020}
}
```


## Built With

* [cnpy](https://github.com/rogersce/cnpy) - Library used to save the data in Numpy format

## Authors

* **Andrea Maiani** <andrea.maiani@nbi.ku.dk> - [skdys](https://github.com/skdys)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
