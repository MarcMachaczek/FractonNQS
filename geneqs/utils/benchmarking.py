import yaml
import global_variables
from tqdm import tqdm
import time
import netket as nk


def parse_settings():
    """
    Load the settings.yml file from the project root directory.
    Returns:
        Nested dictionary containing settings.

    """
    with open(global_variables.SETTINGS_PATH) as f:
        config = yaml.safe_load(f)
    return config


def time_expect(vqs: nk.vqs.VariationalState, operator: nk.operator.AbstractOperator, n_runs: int) -> float:
    """
    Time the calculation of the expected value of an operator using a variational quantum state.
    Args:
        vqs: The variational quantum state.
        operator: The operator to calucalte the expected value of.
        n_runs: Number of times to do the calculation for taking the average time.

    Returns:
        The time required to compute the expected value averaged over n_runs.

    """

    t0 = time.perf_counter()
    for _ in tqdm(range(n_runs), type(operator).__name__):
        vqs.expect(operator)

    return (time.perf_counter() - t0) / n_runs
