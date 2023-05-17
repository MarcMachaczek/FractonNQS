from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(comm.Get_rank())

import jax
import jax.numpy as jnp
import mpi4jax

@jax.jit
def foo(arr):
   arr = arr + rank
   arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
   return arr_sum

a = jnp.zeros((3, 3))
result = foo(a)

if rank == 0:
   print(result)