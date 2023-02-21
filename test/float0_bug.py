import netket as nk
import jax.numpy as jnp
import numpy as np

hilbert = nk.hilbert.Spin(1/2, 2)

ha = nk.operator.LocalOperator(hilbert, dtype=complex)
ha += nk.operator.spin.sigmaz(hilbert, 0)
ha += nk.operator.spin.sigmay(hilbert, 1)

model = nk.models.RBM(alpha=2, param_dtype=float)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=512, dtype=np.int8)

vstate = nk.vqs.MCState(sampler, model, n_samples=1024, chunk_size=256)

optimizer = nk.optimizer.Sgd(learning_rate=0.05)
precon = nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, diag_shift=0.01)
gs = nk.driver.VMC(ha, optimizer, variational_state=vstate, preconditioner=precon)

gs.run(100)

