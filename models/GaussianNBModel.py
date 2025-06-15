from sklearn.naive_bayes import GaussianNB
import numpy as np
from utils.config import RANDOM_STATE

class GaussianNBModel:
    def __init__(self, **kwargs):
        """
        Initialize Gaussian Naive Bayes model
        
        Args:
            **kwargs: Additional parameters for GaussianNB
        """
        self.model = GaussianNB(**kwargs)
    
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

    def get_params(self):
        return self.model.get_params()

    def get_optuna_params(self, trial):
        """Get hyperparameters for Optuna optimization"""
        return {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-9, 1e-5, log=True)
        }

    def get_model_instance(self, params):
        """Create a new model instance with given parameters"""
        return GaussianNB(**params)

    def get_params_from_optuna_params(self, optuna_params):
        """Convert Optuna parameters to model parameters"""
        return optuna_params 