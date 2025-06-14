from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

class QuadraticDiscriminantAnalysisModel:
    def __init__(self, **kwargs):
        """
        Initialize a Quadratic Discriminant Analysis model
        
        Args:
            **kwargs: Additional parameters for QuadraticDiscriminantAnalysis
        """
        self.model = QuadraticDiscriminantAnalysis(**kwargs)
    
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