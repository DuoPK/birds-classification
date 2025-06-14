from sklearn.tree import DecisionTreeClassifier
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import RANDOM_STATE, CV


class DecisionTreeOptimizer(ModelOptimizer):
    def __init__(self, random_state=RANDOM_STATE, scoring='f1_macro', cv=CV):
        self.random_state = random_state

        model = DecisionTreeClassifier(random_state=self.random_state)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
