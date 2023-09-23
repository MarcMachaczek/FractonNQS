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


## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
