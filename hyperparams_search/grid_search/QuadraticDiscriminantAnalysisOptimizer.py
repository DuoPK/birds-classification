from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import CV


class QuadraticDiscriminantAnalysisOptimizer(ModelOptimizer):
    def __init__(self, scoring='f1_macro', cv=CV):
        model = QuadraticDiscriminantAnalysis()
        param_grid = {
            'reg_param': [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
