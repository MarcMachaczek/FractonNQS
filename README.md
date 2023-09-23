# Neural Network Quantum States (NQS) for Fracton Models

This is the code repository for the Master project titled "Neural Network Quantum States for Fracton Models".

Based on JAX and NetKet3, it provides functionality for training NQS on the 2d Toric Code and 3d Checkerboard fracton model in the presence of external magnetic fields. In particular, this package takes handles translational symmetries of qubits and correlators constructed from them to implement a translation-invariant correlation-enhanced RBM for the Toric Code, as described in [Valenti et al.](https://arxiv.org/abs/2103.05017), and the Checkerboard model. Performant operator and neural network implementations with GPU support allow for simulations up to 512 qubits ($L=8$) on the Checkerboard model on a single NVIDIA A100 GPU. By calculating the magnetization from the trained NQS for different magnetic fields, for instance, indications of a strong-first order phase transition are found.

## Setup

A Python$\geq3.10$ installation is required. Further, a Linux based OS is strongly recommended (due to JAX constraints).
After cloning the repository, a new virtual environment `venv` can be created with the command:

```
python -m venv <directory>
```

Anaconda is **not** recommened for this, as described in the [NetKet3 installation guide](https://netket.readthedocs.io/en/latest/docs/install.html). 
Typically, it is easiest to choose `<directory>`=`venv`, which creates the `venv` virtual environment into a newly created `venv` folder of the current working directory (which we assume is the root of the cloned repository).
Then, activate the virtual environment with

```
source venv/bin/activate
```

This codebase currently uses the CUDA 11.8 compatible JAX version. It can be installed into the currently activate environment using the commands

```
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Of course, this requires an NVIDIA GPU with up-to-date drivers. For CPU-only support, install JAX with

```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

For more details refer to the [JAX installation guide](https://github.com/google/jax#installation).

The `requirements.txt` contains all other required packages together with their versions. They can be installed simply by using the command

```
pip install -r requirements.txt
```

This essentially completes the setup process.

However, it is possible to use MPI in order to run NQS optimizations on multiple hosts / GPUs. Assuming a working MPI installation, simply execute 

```
pip install --upgrade "netket[mpi]"
```

to install all required MPI dependencies. For more details on this, refer to the [NetKet3 installation guide](https://netket.readthedocs.io/en/latest/docs/install.html).

## Code structure

The core implementations are located within the `geneqs` directory. Therein, four sub-folders can be found: `models`, `operators`, `sampling`, and `utils`.

- `models` contains all variational wave function implementations. In particular, `rbm.py` containing all RBM-type architectures and `neural_networks.py` containing FFNN-like architectures.

- `operators` folder contains all operator implementations for calculating connected (matrix) elements and some more functionality in a NetKet compatible syntax. This includes `checkerboard.py` and `toric_2d.py`.

- `sampling` folder only contains the implementation of the weighted update rule for MCMC sampling in `update_rules.py`.

- `utils` directory contains many different functionalities that are required for NQS training or implementing symmetries. Most notably, `indexing.py` contains all functionality for indexing qubit positions on the square / cubical lattices, constructing permutations corresponding to translations on the respective lattices for different correlator types etc. Moreover, `training.py` contains our custom training loop, enabling tracking of observables during optimization, progress bars that show the relative time requirement of different steps during NQS optimization and more.

## How to use

## Project status

So far, this project lacks many quality-of-life features that should be mandatory for an accessible and easy-to-use library, like simple toy-examples with proper argument parsing, a detailed documentation, tests and many more. We aim to bring this project to the Python Package Index (PyPI) soon. This will come with a number of improvements like (in rough order):

1. toy-examples on Google Colab for an easy introduction of available functionality
2. clean production scripts with proper argument parsing for easy repoduction of core results
3. documentation
4. benchmarks and tests
5. and more
