from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, C=1.0, max_iter=1000, random_state=42, **kwargs):
        """
        Initialize a Logistic Regression model
        
        Args:
            C (float): Inverse of regularization strength
            max_iter (int): Maximum number of iterations
            random_state (int): Random state for reproducibility
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
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
