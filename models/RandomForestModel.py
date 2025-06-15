from sklearn.ensemble import RandomForestClassifier
from utils.config import RANDOM_STATE


class RandomForestModel:
    def __init__(self, **kwargs):
        default_params = {}
        model_params = {**default_params, **kwargs}
        self.model = RandomForestClassifier(**model_params)

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
            'max_depth': trial.suggest_int('max_depth', 4, 16, step=2),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': RANDOM_STATE
        }

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return RandomForestClassifier(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
