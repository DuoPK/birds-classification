import numpy as np

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, positive_label=1):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.positive_label = positive_label
        self._compute_confusion()

    def _compute_confusion(self):
        self.TP = np.sum((self.y_pred == self.positive_label) & (self.y_true == self.positive_label))
        self.FP = np.sum((self.y_pred == self.positive_label) & (self.y_true != self.positive_label))
        self.FN = np.sum((self.y_pred != self.positive_label) & (self.y_true == self.positive_label))
        self.TN = np.sum((self.y_pred != self.positive_label) & (self.y_true != self.positive_label))

    def confusion_matrix(self):
        """
        Returns the confusion matrix as a 2x2 numpy array:
        [[TN, FP],
         [FN, TP]]
        """
        return np.array([[self.TN, self.FP],
                        [self.FN, self.TP]])

    def accuracy(self):
        """
        Accuracy: (TP + TN)
        """
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total != 0 else 0.0

    def f1_score(self):
        """
        F1 = 2 * TP / (2TP + FP + FN)
        """
        denom = 2 * self.TP + self.FP + self.FN
        return (2 * self.TP) / denom if denom != 0 else 0.0

    def summary(self):
        return {
            "accuracy": self.accuracy(),
            "f1_score": self.f1_score(),
            "confusion_matrix": self.confusion_matrix().tolist()
        }
