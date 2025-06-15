from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class QuadraticDiscriminantAnalysisModel:
    def __init__(self, reg_param=0.0, **kwargs):
        self.model = QuadraticDiscriminantAnalysis(reg_param=reg_param, **kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        reg_param = trial.suggest_float('reg_param', 0.0, 1.0)
        return {'reg_param': reg_param}

    def get_model_instance(self, params):
        return QuadraticDiscriminantAnalysis(**params)

    def get_params_from_optuna_params(self, optuna_params):
        return optuna_params
