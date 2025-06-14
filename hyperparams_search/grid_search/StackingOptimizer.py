from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from training.models.LogisticRegressionModel import LogisticRegressionModel
from training.models.GaussianNBModel import GaussianNBModel
from training.models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import CV


class StackingOptimizer(ModelOptimizer):
    def __init__(self, scoring='f1_macro', cv=CV):
        model = StackingClassifier(
            estimators=[
                ('lr', LogisticRegressionModel().model),
                ('gnb', GaussianNBModel().model),
                ('qda', QuadraticDiscriminantAnalysisModel().model)
            ],
            final_estimator=RandomForestClassifier()
        )
        param_grid = {
            'final_estimator__n_estimators': [50, 100, 200],
            'final_estimator__max_depth': [3, 5, 10],
            'final_estimator__min_samples_split': [2, 5],
            'final_estimator__min_samples_leaf': [1, 2]
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
