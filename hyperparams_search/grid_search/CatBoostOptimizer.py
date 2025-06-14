from catboost import CatBoostClassifier
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import RANDOM_STATE, CV


class CatBoostOptimizer(ModelOptimizer):
    def __init__(self, random_state=RANDOM_STATE, scoring='f1_macro', cv=CV):
        self.random_state = random_state

        model = CatBoostClassifier(
            verbose=0,
            random_state=self.random_state
        )

        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'iterations': [100, 200],
            'l2_leaf_reg': [1, 3, 5]
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
