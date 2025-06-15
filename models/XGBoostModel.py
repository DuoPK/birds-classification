from xgboost import XGBClassifier
from utils.config import RANDOM_STATE


class XGBoostModel:
    def __init__(self, **kwargs):
        default_params = {
            "eval_metric": "logloss"
        }
        model_params = {**default_params, **kwargs}
        self.model = XGBClassifier(**model_params)

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
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss'
        }

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return XGBClassifier(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
