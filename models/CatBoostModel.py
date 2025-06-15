from catboost import CatBoostClassifier
from utils.config import RANDOM_STATE


class CatBoostModel:
    def __init__(self, **kwargs):
        default_params = {
            "verbose": False
        }
        model_params = {**default_params, **kwargs}

        self.model = CatBoostClassifier(**model_params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    # def evaluate(self, X_test, y_test, positive_label=1):
    #     y_pred = self.predict(X_test)
    #     metrics = ClassificationMetrics(y_test, y_pred, positive_label=positive_label)
    #     return metrics.summary()

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        """Get hyperparameters for Optuna optimization"""
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500, step=100),
            'depth': trial.suggest_int('depth', 4, 10, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int('od_wait', 10, 50, step=10),
            'random_state': RANDOM_STATE,
            'verbose': False
        }
        if bootstrap_type == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        return params

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return CatBoostClassifier(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
