from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from models.LogisticRegressionModel import LogisticRegressionModel
from models.GaussianNBModel import GaussianNBModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel

from utils.config import STACKING_MODEL_CV, RANDOM_STATE, N_JOBS


class StackingModel:
    def __init__(self, cv=STACKING_MODEL_CV, lr_params=None, gnb_params=None, qda_params=None,
                 final_estimator_params=None,
                 random_state=RANDOM_STATE, n_jobs=N_JOBS, **kwargs):
        """
        Initialize a Stacking Classifier model
        
        Args:
            cv (int): Number of cross-validation folds
            lr_params (dict): parameters for Logistic Regression
            gnb_params (dict): parameters for Gaussian Naive Bayes
            qda_params (dict): parameters for Quadratic Discriminant Analysis
            final_estimator_params (dict): parameters for final estimator (Random Forest)
            **kwargs: Additional parameters for StackingClassifier
        """
        # Initialize base models with their parameters
        self.base_models = [
            ('lr', LogisticRegressionModel(**(lr_params or {}), random_state=random_state).model),
            ('gnb', GaussianNBModel(**(gnb_params or {})).model),
            ('qda', QuadraticDiscriminantAnalysisModel(**(qda_params or {})).model)
        ]

        # Initialize the final estimator with its parameters
        final_estimator_params = final_estimator_params or {}
        self.final_estimator = RandomForestClassifier(
            random_state=random_state,
            **final_estimator_params
        )

        self.model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.final_estimator,
            cv=cv,
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
        cv = trial.suggest_int('cv', 3, 7)
        lr_params = {'C': trial.suggest_float('lr_C', 0.01, 10.0)}
        gnb_params = {'var_smoothing': trial.suggest_float('gnb_var_smoothing', 1e-10, 1e-8, log=True)}
        qda_params = {'reg_param': trial.suggest_float('qda_reg_param', 0.0, 1.0)}
        final_estimator_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5)
        }
        return {
            'cv': cv,
            'lr_params': lr_params,
            'gnb_params': gnb_params,
            'qda_params': qda_params,
            'final_estimator_params': final_estimator_params
        }

    def get_model_instance(self, params):
        lr = LogisticRegressionModel(**params.get('lr_params', {})).model
        gnb = GaussianNBModel(**params.get('gnb_params', {})).model
        qda = QuadraticDiscriminantAnalysisModel(**params.get('qda_params', {})).model
        final_estimator = RandomForestClassifier(**params.get('final_estimator_params', {}))
        return StackingClassifier(
            estimators=[
                ('lr', lr),
                ('gnb', gnb),
                ('qda', qda)
            ],
            final_estimator=final_estimator,
            cv=params.get('cv', 5)
        )

    def get_params_from_optuna_params(self, optuna_params):
        return optuna_params
