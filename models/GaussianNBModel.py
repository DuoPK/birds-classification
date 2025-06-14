from sklearn.naive_bayes import GaussianNB
import numpy as np

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