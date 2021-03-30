# MEP-GL

Gauged string method for the computation of Minimum Energy Path in Ginzburg-Landau models for superconductors and superfluids.

## Getting Started

Prerequisites:

* CUDA Toolkit
* ZLib
* Boost libraries
* Python scientific libraries (numpy, scipy, matplotlib)

## References

This code has been used for the following papers:

* \[1\] Benfenati A., Maiani A., Rybakov F. N., Bababev E. - *Vortex nucleation barrier revisited* - https://arxiv.org/abs/1911.09513
* \[2\] Maiani A., Benfenati A., Bababev E. - *Vortex nucleation barriers and surface fractional vortices in two-component Ginzburg-Landau model* 

If you use this code please cite as 

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
