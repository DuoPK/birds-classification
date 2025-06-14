from catboost import CatBoostClassifier
from utils.ClassificationMetrics import ClassificationMetrics
from utils.config import RANDOM_STATE


class CatBoostModel:
    def __init__(self, **kwargs):
        default_params = {
            "verbose": False
        }
        model_params = {**default_params, **kwargs}

        self.model = CatBoostClassifier(**model_params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test, positive_label=1):
        y_pred = self.predict(X_test)
        metrics = ClassificationMetrics(y_test, y_pred, positive_label=positive_label)
        return metrics.summary()

    def get_params(self):
        return self.model.get_params()
