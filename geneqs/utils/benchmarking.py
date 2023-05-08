import yaml
import global_variables
from tqdm import tqdm
import time

from typing import Callable


def parse_settings():
    """
    Load the settings.yml file from the project root directory.
    Returns:
        Nested dictionary containing settings.

    """
    with open(global_variables.SETTINGS_PATH) as f:
        config = yaml.safe_load(f)
    return config


def time_function(func: Callable, n_runs: int = 10, *args, **kwargs):
    """
    Measure the average time it takes to execute a provided function over n_runs.

    Args:
        func: Function to benchmark.
        n_runs: How many times the function will be called.
        *args: args for func
        **kwargs: kwargs for func

    Returns:
        The results of the last execution, the average execution time of func over n_runs

    """
    result = None
    t0 = time.perf_counter()
    for _ in tqdm(range(n_runs)):
        result = func(*args, **kwargs)

    return result, round((time.perf_counter() - t0) / n_runs, 4)
