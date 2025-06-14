from enum import Enum, auto
from typing import List

from utils.config import XGBOOST_PARAMS, CATBOOST_PARAMS, MODEL_WITHOUT_N_JOBS_PARAM, SKLEARN_PARAMS


class DatasetType(Enum):
    """Enum for available datasets"""
    MEAN_MINMAX = "mean_minmax.csv"
    # MEDIAN_MINMAX = "median_minmax.csv"
    # MEDIAN_STANDARD = "median_standard.csv"
    #
    # IMPUTE_MINMAX = "impute_minmax.csv"
    # MEAN_STANDARD = "mean_standard.csv"
    # IMPUTE_STANDARD = "impute_standard.csv"
    
    @classmethod
    def get_all_datasets(cls) -> List['DatasetType']:
        """Get all available datasets"""
        return list(cls)
    
    @classmethod
    def get_dataset_path(cls, dataset_type: 'DatasetType') -> str:
        """Get a full path to the dataset"""
        return f"data_generation/data/{dataset_type.value}"
    
    @classmethod
    def get_dataset_name(cls, dataset_type: 'DatasetType') -> str:
        """Get dataset name without extension"""
        return dataset_type.value.replace('.csv', '')

class ModelType(Enum):
    """Enum for available models"""
    SVM = "SVCModel"
    CATBOOST = "CatBoostModel"
    XGBOOST = "XGBoostModel"
    RANDOM_FOREST = "RandomForestModel"
    LOGISTIC_REGRESSION = "LogisticRegressionModel"
    GAUSSIAN_NB = "GaussianNBModel"
    QDA = "QuadraticDiscriminantAnalysisModel"
    VOTING = "VotingModel"
    STACKING = "StackingModel"

    @property
    def base_params(self):
        if self == ModelType.XGBOOST:
            return XGBOOST_PARAMS
        elif self == ModelType.CATBOOST:
            return CATBOOST_PARAMS
        elif self == ModelType.SVM:
            return MODEL_WITHOUT_N_JOBS_PARAM
        elif self in {ModelType.GAUSSIAN_NB, ModelType.QDA}:
            return {}  # No specific params for these models
        else:
            return SKLEARN_PARAMS

    def get_params_from_optuna_params(self, optuna_params: dict):
        """Convert Optuna parameters to model-specific parameters. Only Stacking and Voting params are different."""
        # Delete all params with the 'feature_' prefix
        optuna_params = {k: v for k, v in optuna_params.items() if not k.startswith('feature_')}

        base_params = self.base_params.copy()
        if self == ModelType.STACKING:
            # Optuna parameters for Stacking model:      
            # lr_C
            # lr_max_iter
            # gnb_var_smoothing
            # qda_reg_param
            # rf_n_estimators
            # rf_max_depth
            # class StackingModel:
            #     def __init__(self, cv=CV, lr_params=None, gnb_params=None, qda_params=None,
            #                  final_estimator_params=None,
            #                  random_state=RANDOM_STATE, n_jobs=N_JOBS):
            # self.final_estimator = RandomForestClassifier(
            model_params = {
                'lr_params': {
                    'C': optuna_params.get('lr_C', 1.0),
                    'max_iter': optuna_params.get('lr_max_iter', 100)
                },
                'gnb_params': {
                    'var_smoothing': optuna_params.get('gnb_var_smoothing', 1e-9)
                },
                'qda_params': {
                    'reg_param': optuna_params.get('qda_reg_param', 0.0)
                },
                'final_estimator_params': {
                    'n_estimators': optuna_params.get('rf_n_estimators', 100),
                    'max_depth': optuna_params.get('rf_max_depth', None)
                },
                'cv': optuna_params.get('cv', 5),
            }
        elif self == ModelType.VOTING:
            # Optuna parameters for Voting model:
            # voting_weight_lr
            # voting_weight_gnb
            # voting_weight_qda
            # voting
            model_params = {
                'lr_params': {
                    'C': optuna_params.get('lr_C', 1.0),
                    'max_iter': optuna_params.get('lr_max_iter', 100)
                },
                'gnb_params': {
                    'var_smoothing': optuna_params.get('gnb_var_smoothing', 1e-9)
                },
                'qda_params': {
                    'reg_param': optuna_params.get('qda_reg_param', 0.0)
                },
                'voting': optuna_params.get('voting', 'soft'),
                'weights': [
                    optuna_params.get('voting_weight_lr', 1.0),
                    optuna_params.get('voting_weight_gnb', 1.0),
                    optuna_params.get('voting_weight_qda', 1.0)
                ]
            }
        else:
            # For other models, just return Optuna params
            model_params = optuna_params
        # Merge base params with model-specific params
        return {**base_params, **model_params}
    
    @classmethod
    def get_all_models(cls) -> List['ModelType']:
        """Get all available models"""
        return list(cls)
    
    @classmethod
    def get_model_class_name(cls, model_type: 'ModelType') -> str:
        """Get model class name"""
        return model_type.value 