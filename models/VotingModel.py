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
        """
        Train the model
        
        Args:
            X (numpy.ndarray): Training features
            y (numpy.ndarray): Training labels
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Features to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get probability estimates
        
        Args:
            X (numpy.ndarray): Features to predict probabilities for
            
        Returns:
            numpy.ndarray: Probability estimates
        """
        return self.model.predict_proba(X)
