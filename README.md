# Entropy Density Benchmarking
Analysing the accumulation of Renyi-2 entropy density due to noise in quantum circuits.

## Example code output
Classical simulation of the Renyi-2 entropy density evolution of a 3-qubit hardware-efficient ansatz circuit under local depolarizing noise using the classical shadows protocol for entropy estimation.

![Hardware-efficient circuit under local depolarising noise using Shadows](https://github.com/MDemarty/Entropy_Benchmarking/blob/main/readme_figures/R2d-D0-15-1_M320_K1000_grps5_spls3_random_DP0.008-0.054_AD0-0_meas0-0.png)

## Corresponding article
This is the code used in relation to the article [Entropy Density Benchmarking of Near-Term Quantum Circuits](https://doi.org/10.48550/arXiv.2412.18007) by <ins>Marine Demarty</ins>, James Mills, Kenza Hammam and Raul Garcia-Patron.

## About
This repository provides some basic toolkit for analysing the accumulation of *Renyi-2 entropy density* in a noisy quantum circuit as a function of circuit *depth* (number of layers of gates) and *system size* (number of qubits). 
It is suitable for both quantum circuits run on classical simulations of noisy quantum computers, and those run on actual quantum hardware.

As a work example, we have implemented specific classes of quantum circuits and noise models for the classical simulator, however generalization is possible.
For estimating the entropy, two protocols are implemented: a single-copy NISQ-friendly protocol and a two-copy scalable protocol. 

More details about specific implementations can be found in the next subsections.

### Backend
Quantum hardware
- Superconducting qubits
- Trapped ions

Classical simulator

### Quantum circuits
- Hardware-efficient parameterized quantum circuit with RX, RY, CZ gates (superconducting)
- Hardware-efficient parameterized quantum circuit with GPI, GPI2, MS gates (ion traps)

### Noise model
- Local depolarizing noise with single-qubit and two-qubit gate error probabilities $p_{\text{DP},1}$ and $p_{\text{DP},2}$ respectively
- Amplitude damping noise with single-qubit and two-qubit gate error probabilities $p_{\text{AD},1}$ and $p_{\text{AD},2}$ respectively
- Readout error $p_{\text{meas}}$

### Protocols
Renyi-2 entropy density/Purity estimation protocols that are implemented:
- **Shadows**: Pauli-basis classical shadows protocol [REF](https://www.nature.com/articles/s41567-020-0932-7). This protocol should be preferred for small system sizes, and for devices with low connectivity.
- **SWAP**: SWAP-test-based protocol [REF](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.87.167902). This protocol should be preferred for devices with sufficient connectivity (ideally all-to-all), and is suitable for systems of any sizes. However, note that the measurement circuit of this protocol introduces a higher systematic error in the entropy estimate than the classical shadows protocol due to the presence of two-qubit gates. 

In addition to those protocols, it is possible to run a density matrix simulation of the quantum circuit and obtain the exact Renyi-2 entropy density or purity evolution of its output under a specified noise model:
- **DensMat**: exact density matrix simulation.

## How to use this code
### Setup
`myqiskitenv.yml` contains all necessary dependencies for running this code.

### Workflow for using the entropy toolbox
To obtain the Renyi-2 entopy density evolution using a specific protocol as a function of circuit depth, one needs to run 3 scripts for Shadows or SWAP from the `Code` folder:
- `get_entropy_[protocol]-PROCESS-measurements.py` is used to prepare the output of the target quantum circuit, add the measurement circuit of the specified protocol, and collect measurement outcomes. Those are stored in a csv file.
- `get_entropy_PROCESS-metrics.py` loads the csv file, then implements the classical post-processing of the specified protocol to obtain a Renyi-2 entropy density estimate. This is stored in a json file.
- `get_entropy_PLOT.py` loads the json file, then plots the corresponding entropy density evolution as a function of circuit depth and saves plots in the chosen results folder.

and only 2 scripts for DensMat:
- `get_entropy_[protocol]-PROCESS-metrics.py`is used to prepare the output of the target quantum circuit under some noise model, and obtain its exact Renyi-2 entropy density. This is stored in a json file.
- `get_entropy_[protocol]-PLOT.py` loads the json file, then plots the corresponding entropy density evolution as a function of circuit depth and saves plots in the chosen results folder.

### Workflow for obtaining the plots from our paper
Each script of the form `PAPER....py` corresponds to a single figure from our paper. Figures may be obtained by running each script. Resulting pictures are saved in a `Paper` folder by default.
