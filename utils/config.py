import os
import random
import numpy as np
import torch
import optuna

# Random state configuration
RANDOM_STATE = 42

def set_random_state():
    """Set a random state for all libraries"""
    # Python random
    random.seed(RANDOM_STATE)

    # NumPy
    np.random.seed(RANDOM_STATE)

    # PyTorch
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Grid search configuration
MAX_GRID_SEARCH_COMBINATIONS = 40  # Maximum number of parameter combinations in grid search

CV = 5
TEST_SIZE = 0.2
N_TRIALS = 40
SCORING_OPTUNA = 'f1_score'

SELECT_KBEST = False
K_BEST = 15  # Number of features to select if using SelectKBest

# Optuna feature selection configuration
USE_OPTUNA_FEATURE_SELECTION = False  # Whether to use Optuna for feature selection
MIN_FEATURES_TO_SELECT = 5  # Minimum number of features to select by Optuna
MAX_FEATURES_TO_SELECT = 20  # Maximum number of features to select by Optuna
TIME_SCORE_WEIGHT = None  # Weight for time in Optuna scoring

# Multiprocessing configuration
N_JOBS = os.cpu_count() - 1  # Leave one CPU free for system processes
# N_JOBS = 1

# Sklearn specific settings
SKLEARN_PARAMS = {
    'n_jobs': N_JOBS,
    'random_state': RANDOM_STATE
}

MODEL_WITHOUT_N_JOBS_PARAM = {
    'random_state': RANDOM_STATE
}

# XGBoost specific settings
XGBOOST_PARAMS = {
    'n_jobs': N_JOBS,
    'random_state': RANDOM_STATE,
}

# CatBoost specific settings
CATBOOST_PARAMS = {
    'thread_count': N_JOBS,
    'random_state': RANDOM_STATE,
}

# Optuna specific settings
OPTUNA_PARAMS = {
    'random_state': RANDOM_STATE
}

# Initialize random states
set_random_state()
