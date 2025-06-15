from sklearn.ensemble import VotingClassifier
from models.LogisticRegressionModel import LogisticRegressionModel
from models.GaussianNBModel import GaussianNBModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from utils.config import RANDOM_STATE, N_JOBS


class VotingModel:
    def __init__(self, voting='soft', weights=None, lr_params=None, gnb_params=None, qda_params=None,
                 random_state=RANDOM_STATE, n_jobs=N_JOBS, **kwargs):
        """
        Initialize a Voting Classifier model
        
        Args:
            voting (str): 'hard' or 'soft' voting
            weights (list): weights for each classifier
            lr_params (dict): parameters for Logistic Regression
            gnb_params (dict): parameters for Gaussian Naive Bayes
            qda_params (dict): parameters for Quadratic Discriminant Analysis
            **kwargs: Additional parameters for VotingClassifier
        """
        # Initialize base models with their parameters
        self.base_models = [
            ('lr', LogisticRegressionModel(**(lr_params or {}), random_state=random_state).model),
            ('gnb', GaussianNBModel(**(gnb_params or {})).model),
            ('qda', QuadraticDiscriminantAnalysisModel(**(qda_params or {})).model)
        ]

        self.model = VotingClassifier(
            estimators=self.base_models,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        voting = trial.suggest_categorical('voting', ['soft', 'hard'])
        w1 = trial.suggest_float('weight_lr', 0.5, 2.0)
        w2 = trial.suggest_float('weight_gnb', 0.5, 2.0)
        w3 = trial.suggest_float('weight_qda', 0.5, 2.0)
        lr_params = {'C': trial.suggest_float('lr_C', 0.01, 10.0)}
        gnb_params = {'var_smoothing': trial.suggest_float('gnb_var_smoothing', 1e-10, 1e-8, log=True)}
        qda_params = {'reg_param': trial.suggest_float('qda_reg_param', 0.0, 1.0)}
        return {
            'voting': voting,
            'weights': [w1, w2, w3],
            'lr_params': lr_params,
            'gnb_params': gnb_params,
            'qda_params': qda_params
        }

    def get_model_instance(self, params):
        lr = LogisticRegressionModel(**params.get('lr_params', {})).model
        gnb = GaussianNBModel(**params.get('gnb_params', {})).model
        qda = QuadraticDiscriminantAnalysisModel(**params.get('qda_params', {})).model
        return VotingClassifier(
            estimators=[
                ('lr', lr),
                ('gnb', gnb),
                ('qda', qda)
            ],
            voting=params.get('voting', 'soft'),
            weights=params.get('weights', [1.0, 1.0, 1.0])
        )

    def get_params_from_optuna_params(self, optuna_params):
        return optuna_params
