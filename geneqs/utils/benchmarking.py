import yaml
import global_variables
from tqdm import tqdm
import time
import numpy as np

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
        The results of the last execution, the average execution time of func over n_runs in nanoseconds

    """
    result = None
    times = []
    for _ in tqdm(range(n_runs)):
        t0 = time.perf_counter_ns()
        result = func(*args, **kwargs)
        t1 = time.perf_counter_ns()
        times.append(t1-t0)
        del result
    
    times = np.asarray(times)
    mean, stdev = np.mean(times), np.std(times)

    return round(mean, 4), round(stdev, 4)
