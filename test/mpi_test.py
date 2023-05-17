"""
run this with $mpiexec -n <np> python mpi_test.py
can use env_variable to determine <np> from visible cuda devices, see jupyterhub info
for testing, two gpus are used with 4 cores in total
"""

# %%

from mpi4py import MPI
import os
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(comm.Get_rank())

import jax
import jax.numpy as jnp

# a = jnp.ones(3)

# if jax.distributed.initialize() is not set, jax doesn't see the other processes eg if np==2, jax local devices will be equal to global devices
# if multiple gpus are allocated, not setting jax.distributed.initialize() means single host/process to which all/multiple gpus get assigned

# if it is set and gets info from mpi, then one process is assigned to each GPU controlled by one host, i.p. if np==1 only one gpu is used
# if np > #gpus, then jax falls back to cpu for excess processes

#jax.distributed.initialize()


# print(f"rank {rank} with jax_global_devices: {jax.devices()}")
print(f"rank {rank} with size {size}, jax_local_devices: {jax.local_devices()})")

# %%
