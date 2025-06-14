from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from models.LogisticRegressionModel import LogisticRegressionModel
from models.GaussianNBModel import GaussianNBModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel

from utils.config import CV, RANDOM_STATE, N_JOBS


class StackingModel:
    def __init__(self, cv=CV, lr_params=None, gnb_params=None, qda_params=None,
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