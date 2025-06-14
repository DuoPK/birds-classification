from sklearn.naive_bayes import GaussianNB
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import CV


class GaussianNBOptimizer(ModelOptimizer):
    def __init__(self, scoring='f1_macro', cv=CV):
        model = GaussianNB()
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
