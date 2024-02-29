# Neural Network Quantum States (NQS) for Fracton Models

This is the code repository for the Master project titled "Neural Network Quantum States for Fracton Models".

Based on JAX and NetKet3, it provides functionality for training NQS on the 2d Toric Code and 3d Checkerboard fracton model subject to uniform magnetic fields. In particular, this package handles translational symmetries for qubits and correlators to implement a translation-invariant correlation-enhanced RBM for the Toric Code, as described in [Valenti et al.](https://arxiv.org/abs/2103.05017), and the Checkerboard model. Custom operator and neural network implementations with GPU support allow for simulations of up to 512 qubits on the Checkerboard model using a single NVIDIA A100 or V100 GPU. Calculating the magnetization from the trained NQS for different magnetic fields, for instance, indicates a strong-first order phase transition.

This code repository is still subject to change and represents only a preliminary version.

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

The main production scripts are located within the root directoy of the repository. With `<system>` being either `Checkerboard` or `Toric2D_h`, MPI compatible scripts are provided, `<system>_MPI.py`, as well as scripts that do not require a working MPI installation, `<system>.py`. Moreover, the `<system>_vs_ed.py` scripts are used to compare the performance against exact diagonalization results, as they keep track of the exact ground state energies.

The `<system>_model_comparison.py` scripts implement different neural network architectures to test their performance on the pure systems.

Scripts with the `eval_` prefix can be used to load serialized NQS and calculate observables with them (or other things), possibly with more samples etc.

All scripts with the `plot_` prefix only contain plotting functionality to visualize results from the production scripts. For instance, `plot_model_comparison.py` loads the .json containing training statistics to compare, for instance, the energy of different network architectures during training. The `plot_runstats.py` script creates comparison plots containing the acceptance rates, auto-correlation times and split-$\hat{R}$ values over different field configurations. `plot_tc_observables.py` and `plot_cb_observables.py` create plots containing the energy per spin, magnetization, susceptibility (by finite differences), and V-score from an `*_observables.txt` file, see the **Usage** section for more details.

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

The important **settings for production scripts**, which can be set at the beginning of the corresponding .py file, are the following:

- `swipe` determines the transfer learning protocol. Viable options are `independent`, `left-right`, and `right-left`.
- `field_strengths` is a two-dimensional array that contains all field configurations $(h_x, h_y, h_z)$. 
- `direction_index` is either equal to $0=x$, $1=y$ or $2=z$, and determines along which field compenents the transfer learning order is fixed. For instance, `swipe`=`right-left` and `direction_index`=`1` sorts `field_strengths` in decreasing order of the $y$-field components and transfer learning is then applied starting from the first element of the ordered `field_strengths`.
- `pre_init` determines whether the model is initialized in the exact ground state represenation of the pure system
- `checkpoint` can point towards a serialized NQS (.mpack file) to load that parametrization and chains in order to continure transfer learning / optimization from with this state.


## Project status

So far, this project lacks many quality-of-life features that should be mandatory for an accessible and easy-to-use library, like simple toy-examples with proper argument parsing, a detailed documentation, tests and many more. We aim to bring this project to the Python Package Index (PyPI) soon. This will come with a number of improvements like (in rough order):

1. toy-examples on Google Colab for an easy introduction of available functionality
2. clean production scripts with proper argument parsing for easy repoduction of core results
3. documentation
4. benchmarks and tests
5. ...
