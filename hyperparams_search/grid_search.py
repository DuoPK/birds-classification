from datetime import datetime
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
import time
from utils.ResultsLogger import ResultsLogger
from utils.ClassificationMetrics import ClassificationMetrics
from utils.CrossValidator import CrossValidator
from utils.config import (
    CV, RANDOM_STATE, SCORING_OPTUNA,
    USE_OPTUNA_FEATURE_SELECTION, MIN_FEATURES_TO_SELECT, MAX_FEATURES_TO_SELECT,
    MAX_GRID_SEARCH_COMBINATIONS
)
from hyperparams_search.default_params import get_default_params

class GridSearch:
    def __init__(self, model_class, model_name, input_size=None, output_size=None,
                 cv=CV, scoring=SCORING_OPTUNA, random_state=RANDOM_STATE,
                 log_default_params=True, use_select_kbest=False, k_best=None, feature_names=None,
                 time_score_weight=None, max_combinations=MAX_GRID_SEARCH_COMBINATIONS):
        """
        Parameters:
        - model_class: class of the model to optimize
        - model_name: name of the model (used to get parameter space)
        - input_size: input size for neural network
        - output_size: output size for neural network
        - cv: number of cross-validation folds
        - scoring: metric to optimize ('accuracy' or 'f1_score')
        - random_state: random seed for reproducibility
        - use_select_kbest: whether to use SelectKBest for feature selection
        - k_best: number of features to select if use_select_kbest is True
        - feature_names: list of feature names for feature selection
        - max_combinations: maximum number of parameter combinations to evaluate
        """
        self.model_class = model_class
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.trials_ = []
        self.log_default_params = log_default_params
        self.use_select_kbest = use_select_kbest
        self.k_best = k_best
        self.feature_names = feature_names
        self.time_score_weight = time_score_weight
        self.max_combinations = max_combinations
        
        results_base_dir_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.log_default_params:
            self.results_logger = ResultsLogger(results_base_dir_date)
        else:
            self.results_logger = None
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def _get_param_grid(self):
        """Get parameter grid based on model type, ensuring max combinations limit and including default parameters"""
        if self.model_name == 'NeuralNetworkModel':
            return {
                'hidden_sizes': [[64], [64, 32], [128, 64], [128, 64, 32], [256, 128, 64]],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64],
                'epochs': [50, 100, 150, 200],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
        elif self.model_name == 'SVCModel':
            return {
                'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        elif self.model_name == 'CatBoostModel':
            return {
                'iterations': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            }
        elif self.model_name == 'XGBoostModel':
            return {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'max_depth': [3, 5, 7, 8],
                'min_child_weight': [1, 3, 5]
            }
        elif self.model_name == 'RandomForestModel':
            return {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [4, 8, 12, 16],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [1, 2, 3, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy']
            }
        elif self.model_name == 'LogisticRegressionModel':
            return {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'penalty': ['l2', 'none'],
                'max_iter': [1000]
            }
        elif self.model_name == 'GaussianNBModel':
            return {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        elif self.model_name == 'QuadraticDiscriminantAnalysisModel':
            return {
                'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            }
        elif self.model_name == 'VotingModel':
            # Predefined parameter combinations for base models
            lr_params_list = [
                {'C': 0.1, 'max_iter': 1000},
                {'C': 1.0, 'max_iter': 1000},
                {'C': 10.0, 'max_iter': 1000}
            ]
            gnb_params_list = [
                {'var_smoothing': 1e-9},
                {'var_smoothing': 1e-8}
            ]
            qda_params_list = [
                {'reg_param': 0.0},
                {'reg_param': 0.2}
            ]
            weights_list = [
                [1.0, 1.0, 1.0],
                [1.5, 1.0, 0.5],
                [2.0, 1.0, 0.5],
                [1.0, 2.0, 0.5],
                [1.0, 1.0, 2.0]
            ]
            
            return {
                'voting': ['soft'],
                'weights': weights_list,
                'lr_params': lr_params_list,
                'gnb_params': gnb_params_list,
                'qda_params': qda_params_list
            }
        elif self.model_name == 'StackingModel':
            return {
                'cv': [3, 5, 7],
                'lr_params': [
                    {'C': 0.1, 'max_iter': 1000},
                    {'C': 1.0, 'max_iter': 1000},
                    {'C': 10.0, 'max_iter': 1000}
                ],
                'gnb_params': [
                    {'var_smoothing': 1e-9},
                    {'var_smoothing': 1e-8}
                ],
                'qda_params': [
                    {'reg_param': 0.0},
                    {'reg_param': 0.2}
                ],
                'final_estimator_params': [
                    {
                        'n_estimators': 100,
                        'max_depth': 5,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1
                    },
                    {
                        'n_estimators': 200,
                        'max_depth': 7,
                        'min_samples_split': 3,
                        'min_samples_leaf': 2
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _evaluate_params(self, params, X, y):
        """Evaluate parameters using cross-validation"""
        try:
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Initialize and train the model
                model = self.model_class(**params)
                if hasattr(model, 'train'):
                    model.train(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                # Calculate score using ClassificationMetrics
                metrics = ClassificationMetrics(y_val, y_pred)
                if self.scoring == 'accuracy':
                    score = metrics.accuracy()
                elif self.scoring == 'f1_score':
                    score = metrics.f1_score()
                else:
                    raise ValueError(f"Unknown scoring metric: {self.scoring}")
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Store trial results
            self.trials_.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'status': 'success'
            })
            
            return mean_score
            
        except Exception as e:
            self.logger.error(f"Trial failed with parameters: {params} because of the following error: {str(e)}")
            
            self.trials_.append({
                'params': params,
                'error': str(e),
                'status': 'failed'
            })
            
            return float('-inf')

    def _get_default_params(self):
        """Get default parameters for the model"""
        return get_default_params(self.model_name, self.input_size, self.output_size)

    def _generate_param_combinations(self, param_grid):
        """Generate parameter combinations ensuring max combinations limit and including default parameters"""
        from itertools import product
        
        # Special handling for VotingModel
        if self.model_name == 'VotingModel':
            # Get actual default parameters for the model
            default_params = self._get_default_params()
            
            # Create combinations manually for VotingModel
            param_combinations = []
            for weights in param_grid['weights']:
                for lr_params in param_grid['lr_params']:
                    for gnb_params in param_grid['gnb_params']:
                        for qda_params in param_grid['qda_params']:
                            for voting in param_grid['voting']:
                                params = {
                                    'voting': voting,
                                    'weights': weights,
                                    'lr_params': lr_params,
                                    'gnb_params': gnb_params,
                                    'qda_params': qda_params
                                }
                                param_combinations.append(params)
            
            # If we have more than max_combinations-1 combinations, sample them
            if len(param_combinations) > self.max_combinations - 1:
                rng = np.random.default_rng(self.random_state)
                sampled_combinations = list(rng.choice(param_combinations, size=self.max_combinations - 1, replace=False))
                # Add default parameters as the first combination
                return [default_params] + sampled_combinations
            else:
                # Ensure default parameters are included
                if default_params not in param_combinations:
                    param_combinations.insert(0, default_params)
                return param_combinations
        
        # For other models, use the standard approach
        # Get all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        # Get actual default parameters for the model
        default_params = self._get_default_params()
        
        def convert_param_value(key, value):
            """Convert parameter value to appropriate type"""
            # Special handling for LogisticRegression parameters
            if self.model_name == 'LogisticRegressionModel':
                if key == 'penalty':
                    if value == 'none':
                        return None
                    return str(value)
                elif key == 'solver':
                    return str(value)
                elif key == 'C':
                    return float(value)
                elif key == 'max_iter':
                    return int(value)
            
            # Special handling for StackingModel parameters
            if self.model_name == 'StackingModel':
                if key in ['lr_params', 'gnb_params', 'qda_params', 'final_estimator_params']:
                    return dict(value)  # Convert to Python dict
                elif key == 'cv':
                    return int(value)
            
            # Handle numeric parameters
            if isinstance(value, (int, float, np.integer, np.floating, str, np.str_)):
                # Convert string representations of numbers to the appropriate type
                if isinstance(value, (str, np.str_)):
                    try:
                        value = float(value)
                    except ValueError:
                        return str(value)  # Convert to Python str if not a number
                
                # Convert to the appropriate type based on the parameter name
                if key in ['n_estimators', 'max_depth', 'min_child_weight', 'iterations', 'depth', 
                          'l2_leaf_reg', 'min_samples_split', 'min_samples_leaf', 'max_iter']:
                    return int(value)
                elif key in ['C', 'learning_rate', 'gamma', 'var_smoothing', 'reg_param', 'dropout_rate']:
                    return float(value)
                else:
                    return str(value)  # Convert other string values to Python str
            return value
        
        def is_valid_logistic_regression_params(params):
            """Check if LogisticRegression parameters are valid"""
            solver = params.get('solver')
            penalty = params.get('penalty')
            
            # Map of valid solver-penalty combinations
            valid_combinations = {
                'lbfgs': ['l2', None],
                'newton-cg': ['l2', None],
                'liblinear': ['l1', 'l2'],
                'sag': ['l2', None],
                'saga': ['l1', 'l2', 'elasticnet', None]
            }
            
            return penalty in valid_combinations.get(solver, [])
        
        # If we have more than max_combinations-1 combinations (max_combinations - 1 for default), sample them
        if len(combinations) > self.max_combinations - 1:
            rng = np.random.default_rng(self.random_state)
            sampled_combinations = list(rng.choice(combinations, size=self.max_combinations - 1, replace=False))
            # Add default parameters as the first combination
            param_combinations = [default_params]
            # Convert remaining combinations to dictionaries
            for combination in sampled_combinations:
                params = {key: convert_param_value(key, value) for key, value in zip(keys, combination)}
                # Skip invalid LogisticRegression parameter combinations
                if self.model_name == 'LogisticRegressionModel' and not is_valid_logistic_regression_params(params):
                    continue
                param_combinations.append(params)
        else:
            # Convert all combinations to dictionaries
            param_combinations = []
            for combination in combinations:
                params = {key: convert_param_value(key, value) for key, value in zip(keys, combination)}
                # Skip invalid LogisticRegression parameter combinations
                if self.model_name == 'LogisticRegressionModel' and not is_valid_logistic_regression_params(params):
                    continue
                param_combinations.append(params)
            # Ensure default parameters are included
            if default_params not in param_combinations:
                param_combinations.insert(0, default_params)
        
        return param_combinations

    def fit(self, X, y, dataset_name):
        """Find the best parameters using grid search"""
        # Get parameter grid
        param_grid = self._get_param_grid()
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Evaluate each parameter combination
        best_score = float('-inf')
        best_params = None
        
        for params in param_combinations:
            score = self._evaluate_params(params, X, y)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        # Log results
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self 