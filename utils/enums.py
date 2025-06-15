from enum import Enum
from pathlib import Path
from typing import List, Self

from models.RandomForestModel import RandomForestModel
from models.SVCModel import SVCModel
from models.CatBoostModel import CatBoostModel
from models.XGBoostModel import XGBoostModel
from models.LogisticRegressionModel import LogisticRegressionModel
from models.GaussianNBModel import GaussianNBModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from models.VotingModel import VotingModel
from models.StackingModel import StackingModel

MODEL_CLASS_MAP = {
    "XGBoostModel": XGBoostModel,
    "SVCModel": SVCModel,
    "CatBoostModel": CatBoostModel,
    "RandomForestModel": RandomForestModel,
    "LogisticRegressionModel": LogisticRegressionModel,
    "GaussianNBModel": GaussianNBModel,
    "QuadraticDiscriminantAnalysisModel": QuadraticDiscriminantAnalysisModel,
    "VotingModel": VotingModel,
    "StackingModel": StackingModel,
}


class DatasetType(Enum):
    """Enum for available datasets"""
    MY_AUDIO = "mfcc13_seg512ms_dist400ms_maxseg10_bp1000-8000Hz_order5_peak0.1_prom0.1.csv"

    # AUDIO_MFCC_SAVGOL = "savgol_mfcc.csv"

    @classmethod
    def get_all_datasets(cls) -> list[str]:
        """Get all available datasets"""
        return [e.value for e in cls]

    @classmethod
    def get_all_csv_paths(cls, base_dir: str = "data_generation/data") -> List[str]:
        """Returns a list of all .csv files in the subdirectories of base_dir"""
        return [str(p) for p in Path(base_dir).rglob("*.csv")]

    @classmethod
    def get_dataset_path(cls, dataset_type: Self | str) -> str:
        """Returns the full path to the dataset"""
        # Searches for a file in subdirectories
        base_dir = Path("data_generation/data")
        if isinstance(dataset_type, DatasetType):
            dataset_type = dataset_type.value
        for p in base_dir.rglob(dataset_type):
            return str(p)
        raise FileNotFoundError(f"File not found {dataset_type} w {base_dir}")

    @classmethod
    def get_dataset_name(cls, dataset_type: Self | str) -> str:
        """Get dataset name without extension"""
        if isinstance(dataset_type, DatasetType):
            dataset_type = dataset_type.value
        return dataset_type.replace('.csv', '')


class ModelType(Enum):
    """Enum for available models"""
    XGBOOST = "XGBoostModel"
    SVM = "SVCModel"
    CATBOOST = "CatBoostModel"
    RANDOM_FOREST = "RandomForestModel"
    LOGISTIC_REGRESSION = "LogisticRegressionModel"
    GAUSSIAN_NB = "GaussianNBModel"
    QDA = "QuadraticDiscriminantAnalysisModel"
    VOTING = "VotingModel"
    STACKING = "StackingModel"

    @classmethod
    def get_all_models(cls) -> List['ModelType']:
        """Get all available models"""
        return list(cls)

    @classmethod
    def get_model_class_name(cls, model_type: 'ModelType') -> str:
        """Get model class name"""
        return model_type.value


def get_model_instance(model_type: ModelType):
    class_name = model_type.value
    model_class = MODEL_CLASS_MAP.get(class_name)
    if model_class is None:
        raise ValueError(f"Unknown model type: {class_name}")
    return model_class()
