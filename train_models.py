import os
import warnings

import pandas as pd
import time
import json
import logging
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif

from models.GaussianNBModel import GaussianNBModel
from models.LogisticRegressionModel import LogisticRegressionModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from models.RandomForestModel import RandomForestModel
from models.SVCModel import SVCModel
from models.StackingModel import StackingModel
from models.VotingModel import VotingModel
from utils.ClassificationMetrics import ClassificationMetrics
from utils.StratifiedTrainTestSplitter import StratifiedTrainTestSplitter
from hyperparams_search.optuna_search import OptunaSearch
from utils.enums import DatasetType, ModelType
from utils.config import (
    RANDOM_STATE, USE_OPTUNA_FEATURE_SELECTION, OPTUNA_PARAMS, set_random_state, CV, N_TRIALS, TEST_SIZE, SELECT_KBEST,
    K_BEST, TIME_SCORE_WEIGHT
)

# Import all models
from models.CatBoostModel import CatBoostModel
from models.XGBoostModel import XGBoostModel
from scipy import linalg

warnings.filterwarnings("ignore", category=linalg.LinAlgWarning)
datetime_dir_name = datetime.now().strftime('%Y%m%d_%H%M%S')

# Model mapping
MODEL_CLASSES = {
    ModelType.CATBOOST: CatBoostModel,
    ModelType.XGBOOST: XGBoostModel,
    ModelType.RANDOM_FOREST: RandomForestModel,
    ModelType.SVM: SVCModel,
    ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
    ModelType.GAUSSIAN_NB: GaussianNBModel,
    ModelType.QDA: QuadraticDiscriminantAnalysisModel,
    ModelType.VOTING: VotingModel,
    ModelType.STACKING: StackingModel
}

def setup_logging(dataset_name, model_name):
    log_dir = "training/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{dataset_name}_{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Diagnosis', axis=1).values
    y = df['Diagnosis'].values
    feature_names = df.drop('Diagnosis', axis=1).columns.tolist()
    return X, y, feature_names

def get_model_class(model_type: ModelType):
    """Get a model class from the model type"""
    return MODEL_CLASSES[model_type]

def train_and_evaluate_model(model_type: ModelType, dataset_type: DatasetType, use_select_kbest=False, k_best=None):
    # Setup logging
    dataset_name = DatasetType.get_dataset_name(dataset_type)
    model_name = ModelType.get_model_class_name(model_type)
    logger = setup_logging(dataset_name, model_name)
    
    logger.info(f"Starting training for {model_name} on {dataset_name}")
    if use_select_kbest:
        logger.info(f"Using SelectKBest with k={k_best}")
    
    # Load data
    dataset_path = DatasetType.get_dataset_path(dataset_type)
    X, y, feature_names = load_dataset(dataset_path)
    
    # Apply SelectKBest if enabled
    if use_select_kbest:
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        feature_names = [feature_names[i] for i in selected_indices]
        logger.info(f"Selected {k_best} best features using SelectKBest: {feature_names}")
    
    # Split into train and test sets using custom StratifiedTrainTestSplitter
    splitter = StratifiedTrainTestSplitter(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    # Initialize results dictionary
    results = {
        "dataset": dataset_name,
        "model": model_name,
        "feature_selection": {
            "used": use_select_kbest,
            "k_best": k_best if use_select_kbest else None,
            "feature_names": feature_names
        },
        "hyperparameter_search": {},
        "custom_cv_results": {},
        "sklearn_cv_results": {},
        "final_test_results": {}
    }
    
    logger.info("Starting hyperparameter search with Optuna")
    model_class = get_model_class(model_type)
    
    # Initialize Optuna search with multiprocessing
    optuna_search = OptunaSearch(
        model_class=model_class,
        model_name=model_name,
        n_trials=N_TRIALS,
        cv=CV,
        scoring='f1_score',
        **OPTUNA_PARAMS,
        log_default_params=False,
        use_select_kbest=use_select_kbest,
        k_best=k_best,
        feature_names=feature_names,
        time_score_weight=TIME_SCORE_WEIGHT
    )
    
    # Run hyperparameter optimization
    optuna_search.fit(X_train, y_train, dataset_name)
    
    results["hyperparameter_search"] = {
        "best_params": optuna_search.best_params_,
        "best_score": optuna_search.best_score_,
        "trials": optuna_search.trials_
    }
    
    # Final training and testing with the best parameters
    logger.info("Starting final training and testing with best parameters")
    
    model_best_params = optuna_search.best_params_.copy()
    
    # Apply feature selection if enabled
    if USE_OPTUNA_FEATURE_SELECTION:
        # Get selected feature indices
        selected_indices = []
        for i, feature in enumerate(feature_names):
            param_name = f'feature_{feature}'
            if param_name in model_best_params and model_best_params[param_name] == 1:
                selected_indices.append(i)
                # Remove feature selection parameter
                del model_best_params[param_name]
        
        # Apply selection to test data
        X_test = X_test[:, selected_indices]
        X_train = X_train[:, selected_indices]
    
    model_all_params = model_type.get_params_from_optuna_params(model_best_params)
    
    final_model = model_class(**model_all_params)
    final_train_start = time.time()
    if hasattr(final_model, 'train'):
        final_model.train(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    metrics = ClassificationMetrics(y_test, y_pred)
    
    final_train_time = time.time() - final_train_start
    results["final_test_results"] = {
        "metrics": metrics.summary(),
        "training_time": final_train_time
    }
    
    # Save results
    base_results_dir = "training/results"
    subcatalog = "optuna"
    if USE_OPTUNA_FEATURE_SELECTION:
        subcatalog = f"{subcatalog}_select-features-by-optuna"
    if use_select_kbest:
        subcatalog = f"{subcatalog}_select-kbest/{k_best}-best"
    results_dir = f"{base_results_dir}/{subcatalog}/{datetime_dir_name}"
    
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Training completed. Results saved to {results_file}")
    return results

def main():
    # Set a random state for all libraries
    set_random_state()
    
    # Train each model on each dataset
    for dataset_type in DatasetType.get_all_datasets():
        for model_type in MODEL_CLASSES.keys():
            try:
                train_and_evaluate_model(model_type, dataset_type, SELECT_KBEST, K_BEST)
            except Exception as e:
                logging.error(f"Error training {model_type} on {dataset_type}: {str(e)}")

if __name__ == "__main__":
    main()
    