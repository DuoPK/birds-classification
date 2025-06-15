import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)
import logging


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, positive_label=None):
        """
        Initialize ClassificationMetrics with true and predicted labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_label: Optional label for binary classification metrics
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.positive_label = positive_label
        self.labels = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.labels)
        
        # Log dataset information
        logging.info(f"Number of classes: {self.n_classes}")
        logging.info(f"Unique labels: {self.labels}")
        logging.info(f"Class distribution in true labels: {dict(zip(*np.unique(self.y_true, return_counts=True)))}")
        logging.info(f"Class distribution in predictions: {dict(zip(*np.unique(self.y_pred, return_counts=True)))}")

    def accuracy(self):
        """Calculate accuracy score"""
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self):
        """Calculate F1 score"""
        if self.n_classes == 2 and self.positive_label is not None:
            return f1_score(self.y_true, self.y_pred, pos_label=self.positive_label)
        return f1_score(self.y_true, self.y_pred, average='weighted')

    def precision(self):
        """Calculate precision score"""
        if self.n_classes == 2 and self.positive_label is not None:
            return precision_score(self.y_true, self.y_pred, pos_label=self.positive_label)
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def recall(self):
        """Calculate recall score"""
        if self.n_classes == 2 and self.positive_label is not None:
            return recall_score(self.y_true, self.y_pred, pos_label=self.positive_label)
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def confusion_matrix(self):
        """Calculate confusion matrix"""
        return confusion_matrix(self.y_true, self.y_pred, labels=self.labels)

    def roc_auc(self, y_pred_proba):
        """Calculate ROC AUC score"""
        if self.n_classes == 2:
            return roc_auc_score(self.y_true, y_pred_proba[:, 1])
        return roc_auc_score(self.y_true, y_pred_proba, multi_class='ovr')

    def summary(self):
        """Get a summary of all metrics"""
        try:
            return {
                'accuracy': self.accuracy(),
                'f1_score': self.f1_score(),
                'precision': self.precision(),
                'recall': self.recall(),
                'confusion_matrix': self.confusion_matrix(),
                'labels': self.labels,
                'n_classes': self.n_classes
            }
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            return {
                'error': str(e),
                'labels': self.labels,
                'n_classes': self.n_classes
            }
