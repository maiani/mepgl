# MEPGL

Gauged string method for the computation of minimum energy paths (MEP) in Ginzburg-Landau (GL) models for superconductors and superfluids.

<img src="cover.gif" width="100%" />

This code is able to compute minimum energy paths of system described by the free energy

<img src="https://render.githubusercontent.com/render/math?math=F[\mathbf{A}, \psi_1, \psi_2] = \sum_{\alpha=1,2} [ \sum_{k=x,y} \frac{|D_{k} \psi_\alpha|^2}{2 m_{\alpha,kk}} %2B \frac{b_\alpha}{2} ( \frac{a_\alpha}{b_\alpha} %2B |\psi_\alpha|^2)^2] %2B \frac{1}{2}(\nabla \times \mathbf{A} - \mathbf{H})^2 %2B V_{\mathrm{int}}(\psi_1, \psi_2)\,.">

It is possible to simulate any 2D geometry with an arbitrary initial guess. Both single component system, i.e conventional superconductors, and unconventional multicomponent systems can be simulated. This includes anisotropic materials and systems featuring condensates with different charges. 

Linear Josephson coupling, density coupling and biquadratic Josephson coupling are implemented in the direct interaction term:

<img src="https://render.githubusercontent.com/render/math?math=V_{\mathrm{int}}(\psi_1, \psi_2) = \frac{\eta}{2}(\psi_1 \psi_2^* %2B \mathrm{c.c.}) %2B \frac{\gamma}{2} |\psi_1|^2 |\psi_2|^2 %2B \frac{\delta}{4} ([ \psi_1 \psi_2^* ]^2 %2B \mathrm{c.c.})\,.">

The code can also simulate thin films, where demagnetizing fields are ignored.

## Getting Started

### Prerequisites:
To run this software, you need

* CUDA Toolkit
* ZLib
* Boost libraries
* Python scientific libraries (NumPy, SciPy, matplotlib)

A conda yml file is provided to get these libraries, except for the CUDA toolkit.
To use it, run 
``` 
conda env create -f environment.yml
conda activate mepgl
```

You need to activate the `mepgl` environment in each session, before running the code.

### Setting precision
To set the numeric precision, you need to edit `./src/real.cuh` and `./mepgl_lib/launcher_utils.py`

### Writing down a config file
The first step to run the program is to edit the configuration file `./config.py`. 
This file provides the parameters of the system and the information to build the initial guess for the path.
Some example configuration files are included in the `examples` folder.

### Run the program

Once the config file is set up, just run the simulation with `./run.py`. 
If you need to run it in debug mode: `./run.py --debug` or `./run.py --d`
If you want to skip the generation of the initial guess: `./run.py --no-init` or `./run.py -ni` 
If you want to reload the output of the simulation and continue the optimization: `./run.py --reload-continue` or `./run.py -rc` 

Runtime commands (press the key and then Enter):
- `+` and `-` change the number of relaxation steps.
- `q` stops the simulation and saves the results.
- `C` switch on and off NLCG.

The results of the simulation can be found in the `./simulations` folder under a directory with the name of the simulation.

To run simulations in batches the `./batch_run.py` file can be used. You will need to edit it and the config file.

If you want to continue a simulation run `./reload.py` to load the output of the previous run as initial guess, 
then run the simulation with `./run.py --noinit` to prevent overwriting the input files.

You can check the mep in real-time while the simulation is running using `./rtanim.py`. 

## References
This code has been used for the following papers:

* \[1\] Benfenati A., Maiani A., Rybakov F. N., Bababev E. - *Vortex nucleation barrier revisited* - https://arxiv.org/abs/1911.09513
* \[2\] Maiani A., Benfenati A., Bababev E. - *Vortex nucleation barriers and stable fractional vortices near boundaries in multicomponent superconductors* - https://arxiv.org/abs/2111.01061

If you use this code, please cite the repository and the methods described in 
```
@article{Benfenati_PRB_2020,
author = {Benfenati, Andrea and Maiani, Andrea and Rybakov, Filipp N. and Babaev, Egor},
doi = {10.1103/PhysRevB.101.220505},
issn = {24699969},
journal = {Physical Review B},
month = {jun},
number = {22},
pages = {220505},
title = {{Vortex nucleation barrier in superconductors beyond the Bean-Livingston approximation: A numerical approach for the sphaleron problem in a gauge theory}},
url = {http://arxiv.org/abs/1911.09513 https://link.aps.org/doi/10.1103/PhysRevB.101.220505},
volume = {101},
year = {2020}
}
```
and, for the multicomponent version, in 
```
@article{Maiani_PRB_2022, 
year = {2022}, 
title = {{Vortex nucleation barriers and stable fractional vortices near boundaries in multicomponent superconductors}}, 
author = {Maiani, Andrea and Benfenati, Andrea and Babaev, Egor}, 
journal = {Physical Review B}, 
issn = {2469-9950}, 
doi = {10.1103/PhysRevB.105.224507}, 
pages = {224507}, 
number = {22}, 
volume = {105}, 
}
```

## Built With

* [cnpy](https://github.com/rogersce/cnpy) - Library used to save the data in Numpy format

## Authors

* **Andrea Maiani** <andrea.maiani@nbi.ku.dk> - [maiani](https://github.com/maiani)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
