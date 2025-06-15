from sklearn.svm import SVC
from utils.config import RANDOM_STATE


class SVCModel:
    def __init__(self, **kwargs):
        default_params = {}
        model_params = {**default_params, **kwargs}
        self.model = SVC(**model_params, probability=True)

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
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': RANDOM_STATE
        }

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return SVC(**params, probability=True)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
