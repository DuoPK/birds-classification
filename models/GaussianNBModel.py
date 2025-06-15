from sklearn.naive_bayes import GaussianNB


class GaussianNBModel:
    def __init__(self, **kwargs):
        self.model = GaussianNB(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        return {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-9, 1e-5, log=True)
        }

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return GaussianNB(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params
