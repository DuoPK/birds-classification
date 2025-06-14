from sklearn.ensemble import VotingClassifier
from training.models.LogisticRegressionModel import LogisticRegressionModel
from training.models.GaussianNBModel import GaussianNBModel
from training.models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import CV


class VotingOptimizer(ModelOptimizer):
    def __init__(self, scoring='f1_macro', cv=CV):
        model = VotingClassifier(
            estimators=[
                ('lr', LogisticRegressionModel().model),
                ('gnb', GaussianNBModel().model),
                ('qda', QuadraticDiscriminantAnalysisModel().model)
            ],
            voting='soft'
        )
        param_grid = {
            'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
            'voting': ['hard', 'soft']
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
