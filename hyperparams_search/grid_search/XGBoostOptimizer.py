from xgboost import XGBClassifier
from training.hyperparams_search.grid_search.ModelOptimizer import ModelOptimizer
from training.utils.config import RANDOM_STATE, CV


class XGBoostOptimizer(ModelOptimizer):
    def __init__(self, random_state=RANDOM_STATE, scoring='f1_macro', cv=CV):
        self.random_state = random_state

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=self.random_state
        )

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.01],
            'reg_lambda': [1.0, 10.0]
        }

        super().__init__(model=model, param_grid=param_grid, scoring=scoring, cv=cv)
