from sklearn import clone
from sklearn.model_selection import GridSearchCV

from training.utils.config import CV


class ModelOptimizer:
    def __init__(self, model, param_grid, scoring='f1_macro', cv=CV):
        """
        Parameters:
        - model: a preconfigured sklearn-compatible model instance (e.g., DecisionTreeClassifier())
        - param_grid: dictionary of hyperparameters to search
        - scoring: metric to optimize (e.g. 'f1_macro', 'accuracy')
        - cv: number of folds for cross-validation
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv

    def optimize(self, X_train, y_train):
        model_copy = clone(self.model)
        grid = GridSearchCV(model_copy, self.param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        grid.fit(X_train, y_train)

        return grid.best_estimator_, grid.best_params_, grid.best_score_
