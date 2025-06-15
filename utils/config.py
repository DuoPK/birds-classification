import os
import random
import numpy as np

STACKING_MODEL_CV = 5
N_JOBS = os.cpu_count() - 1  # Leave one CPU free for system processes
RANDOM_STATE = 42


def set_random_state():
    """Set a random state for all libraries"""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


set_random_state()
