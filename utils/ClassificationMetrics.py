import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = np.array([0, 1, 2, 3])
        self._compute_confusion()

    def _compute_confusion(self):
        self.cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)

    def confusion_matrix(self):
        return self.cm

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='weighted', labels=self.labels)

    def summary(self):
        return {
            "accuracy": self.accuracy(),
            "f1_score": self.f1_score(),
            "confusion_matrix": self.confusion_matrix().tolist()
        }
