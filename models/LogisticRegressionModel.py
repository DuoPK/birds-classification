from sklearn.linear_model import LogisticRegression
from utils.config import RANDOM_STATE
import optuna


class LogisticRegressionModel:
    def __init__(self, C=1.0, max_iter=1000, random_state=RANDOM_STATE, **kwargs):
        """
        Initialize a Logistic Regression model
        
        Args:
            C (float): Inverse of regularization strength
            max_iter (int): Maximum number of iterations
            random_state (int): Random state for reproducibility
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        """Get hyperparameters for Optuna optimization"""
        solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 2000, step=100)

        # Map of correct solver-penalty combinations
        valid_combinations = {
            'lbfgs': ['l2', None],
            'newton-cg': ['l2', None],
            'sag': ['l2', None],
            'saga': ['l1', 'l2', 'elasticnet', None],
        }

        if penalty not in valid_combinations.get(solver, []):
            raise optuna.TrialPruned()  # reject an invalid combination

        params = {
            'solver': solver,
            'penalty': penalty,
            'class_weight': class_weight,
            'C': C,
            'max_iter': max_iter,
            'random_state': RANDOM_STATE
        }

        # For elasticnet l1_ratio is required
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        return params

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return LogisticRegression(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
