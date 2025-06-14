from sklearn.linear_model import LogisticRegression
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import CV


class LogisticRegressionOptimizer(ModelOptimizer):
    def __init__(self, scoring='f1_macro', cv=CV):
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
