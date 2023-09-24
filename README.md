# Neural Network Quantum States (NQS) for Fracton Models

This is the code repository for the Master project titled "Neural Network Quantum States for Fracton Models".

Based on JAX and NetKet3, it provides functionality for training NQS on the 2d Toric Code and 3d Checkerboard fracton model in the presence of external magnetic fields. In particular, this package handles translational symmetries for qubits and correlators constructed from them to implement a translation-invariant correlation-enhanced RBM for the Toric Code, as described in [Valenti et al.](https://arxiv.org/abs/2103.05017), and the Checkerboard model. Performant operator and neural network implementations with GPU support allow for simulations up to 512 qubits ($L=8$) on the Checkerboard model on a single NVIDIA A100 GPU. Calculating the magnetization from the trained NQS for different magnetic fields, for instance, indicates a strong-first order phase transition.

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

- `models` contains all variational wave function implementations. In particular, `rbm.py` contains all RBM-type architectures and `neural_networks.py` contains FFNN-like architectures.

- `operators` contains all operator implementations for calculating connected (matrix) elements and some more functionality in a NetKet compatible syntax. This includes `checkerboard.py` and `toric_2d.py`.

- `sampling` so far implements only the implementation of the weighted update rule for MCMC sampling in `update_rules.py`.

- `utils` contains many different functionalities that are required for NQS training and implementing symmetries. Most notably, `indexing.py` contains all functionality for indexing qubit positions on the square / cubical lattices, constructing permutations corresponding to translations on the respective lattices for different correlator types etc. Moreover, `training.py` contains a custom training loop, enabling tracking of observables during optimization, a progress bar that shows the relative time requirement of different steps during NQS optimization and more.

The main production scripts are located within the root directoy of the repository. With `<system>` being either `Checkerboard` or `Toric2D_h`, MPI compatible scripts, `<system>_MPI.py`, and scripts that do not require a working MPI installation, `<system>.py`, are provided.

The `<system>_vs_ed.py` scripts are used to compare the performance against exact diagonalization results.

The `<system>_model_comparison.py` scripts implement different neural network architectures to test their performance on the pure systems.

## Usage

To run an NQS optimization, simply use the command

```
python <script_name>.py
```

from within the activated `venv` environment. 
For NQS training distributed over multiple hosts, run the command:

```
mpiexec -np <number_of_hosts> python <system>_MPI.py
```

The `<number_of_hosts>` must not exceed the number of available GPUs.

By default, all these scripts produce 4 files for each field configuration into the `save_path` (by default, the `checkerboard` or `toric2d_h` folder in the results directory):

- a .pdf file that contains most hyperparameter settings and shows the energy vs training iterations
- a .txt file with prefix `params_` that contains all optimized parameters in a “human-readable” form
- a .json file with the prefix `stats_` that contains all training stats like energy variances, split-$\hat{R}$ values, auto-correlation times etc.
- an .mpack file with the prefix `vqs_` that serializes the final trained model, including all parameters and the states of the Markov chains, which can be used to reconstruct the trained model for evaluation purposes

Moreover, an `*_observables.txt` file is created that contains the energy, magnetization and absolute magnetization together with errors and variances for all field_strengths. This allows for a quick evaluation of results.


## Project status

So far, this project lacks many quality-of-life features that should be mandatory for an accessible and easy-to-use library, like simple toy-examples with proper argument parsing, a detailed documentation, tests and many more. We aim to bring this project to the Python Package Index (PyPI) soon. This will come with a number of improvements like (in rough order):

1. toy-examples on Google Colab for an easy introduction of available functionality
2. clean production scripts with proper argument parsing for easy repoduction of core results
3. documentation
4. benchmarks and tests
5. ...
