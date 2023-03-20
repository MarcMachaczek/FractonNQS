from scipy.sparse import csr_matrix as _csr_matrix
import netket as nk
import numpy as np


def operator_to_sparse(operator: nk.operator.DiscreteOperator):
    hilbert = operator.hilbert

    x = hilbert.all_states()

    sections = np.empty(x.shape[0], dtype=np.int32)
    x_prime, mels = operator.get_conn_flattened(x, sections)

    numbers = hilbert.states_to_numbers(np.asarray(np.split(x_prime, sections, axis=-1)[:-1]))

    sections1 = np.empty(sections.size + 1, dtype=np.int32)
    sections1[1:] = sections
    sections1[0] = 0

    return _csr_matrix(
        (mels, numbers, sections1),
        shape=(hilbert.n_states, hilbert.n_states),
    )
