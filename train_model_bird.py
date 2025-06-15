import os
from pathlib import Path

import pandas as pd
import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import joblib
import seaborn as sns

from hyperparams_search.default_params import get_default_params
from utils.config import RANDOM_STATE
from utils.enums import ModelType, DatasetType, get_model_instance
from sklearn.feature_selection import SelectKBest, f_classif

from utils.utils import load_dataset, split_data_by_source, find_csv_paths

TEST_SIZE = 0.2
CV = 5
N_TRIALS = 2
OPTUNA_TIMEOUT = 300  # 5 minutes
SCORING_OPTUNA = 'f1_weighted'
SELECT_KBEST = True
K_BEST = 10

MODELS_DIR = "models/saved"
RESULTS_DIR = "results"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def train_and_evaluate_model(model_type: ModelType, dataset_filename: str, use_select_kbest=SELECT_KBEST,
                             k_best=K_BEST):
    """Train and evaluate model using scikit-learn's cross-validation"""
    try:
        logging.info(f"Starting training for {model_type.value} on {dataset_filename}")

        # Load dataset
        dataset_path = DatasetType.get_dataset_path(dataset_filename)  # ZMIANA
        df, feature_cols = load_dataset(dataset_path)  # ZMIANA
        logging.info(f"Loaded dataset with {len(df)} samples and {len(feature_cols)} features")

        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = split_data_by_source(
            df, feature_cols, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logging.info(f"Split data into {len(train_files)} training files and {len(test_files)} test files")

        # Convert text labels to numeric values
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # SelectKBest
        if use_select_kbest:
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_train = selector.fit_transform(X_train, y_train_encoded)
            X_test = selector.transform(X_test)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_cols[i] for i in selected_indices]
            logging.info(f"Selected {k_best} best features: {selected_features}")

        # Initialize model
        model = get_model_instance(model_type)

        dataset_name = DatasetType.get_dataset_name(dataset_filename)
        study_name = f'{model_type.value}_{dataset_name}_kbest-{k_best}' if use_select_kbest else f'{model_type.value}_{dataset_name}'
        # Perform hyperparameter optimization
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            load_if_exists=True,
            storage='sqlite:///optuna_study_sound.db',
        )

        if len(study.trials) == 0:
            default_params = get_default_params(model_type.value)
            study.enqueue_trial(default_params)

        def objective(trial):
            params = model.get_optuna_params(trial)
            model_instance = model.get_model_instance(params)
            scores = cross_val_score(
                model_instance, X_train, y_train_encoded,
                cv=CV, scoring=SCORING_OPTUNA
            )  # Using stratified cross-validation
            return scores.mean()

        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

        # Train final model with best parameters
        best_params = model.get_params_from_optuna_params(study.best_params)
        final_model = model.get_model_instance(best_params)
        final_model.fit(X_train, y_train_encoded)

        # Evaluate on a test set
        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)

        # Calculate metrics
        n_classes = len(np.unique(y_test_encoded))
        if n_classes > 2:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])

        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'f1': f1_score(y_test_encoded, y_pred, average='weighted'),
            'precision': precision_score(y_test_encoded, y_pred, average='weighted'),
            'recall': recall_score(y_test_encoded, y_pred, average='weighted'),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred)
        }

        # Save metrics to CSV
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_file = f"{model_type.value}_{dataset_name}_results.csv"
        if use_select_kbest:
            results_file = f"{model_type.value}_{dataset_name}_kbest-{k_best}_results.csv"
        results_path = os.path.join(RESULTS_DIR, results_file)

        # Convert metrics to flat dictionary for CSV
        flat_metrics = {
            'model_type': model_type.value,
            'dataset': dataset_filename,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': str(metrics['confusion_matrix'].tolist()),  # Convert confusion matrix to string
            'best_params': str(best_params),  # Convert dict to string for CSV
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to CSV
        df = pd.DataFrame([flat_metrics])
        df.to_csv(results_path, index=False)

        # Save confusion matrix separately as CSV
        cm_path_file = f"{model_type.value}_{dataset_name}_confusion_matrix.csv"
        if use_select_kbest:
            cm_path_file = f"{model_type.value}_{dataset_name}_kbest-{k_best}_confusion_matrix.csv"
        cm_path = os.path.join(RESULTS_DIR, cm_path_file)
        pd.DataFrame(metrics['confusion_matrix']).to_csv(cm_path, index=False)

        class_labels = label_encoder.classes_
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"Confusion Matrix: {model_type.value} na {dataset_name}")
        plt.xlabel("Prediction")
        plt.ylabel("Real Label")
        img_path_file = f"{model_type.value}_{dataset_name}_confusion_matrix.png"
        if use_select_kbest:
            img_path_file = f"{model_type.value}_{dataset_name}_kbest-{k_best}_confusion_matrix.png"
        img_path = os.path.join(RESULTS_DIR, img_path_file)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        logging.info(f"Confucion-matrix img saved: {img_path}")

        # Save model and results
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path_file = f"{model_type.value}_{dataset_name}.joblib"
        if use_select_kbest:
            model_path_file = f"{model_type.value}_{dataset_name}_kbest-{k_best}.joblib"
        model_path = os.path.join(MODELS_DIR, model_path_file)
        joblib.dump(final_model, model_path)

        # Log results
        logging.info(f"Training completed for {model_type.value}")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Test metrics: {metrics}")
        logging.info(f"Model saved to: {model_path}")
        logging.info(f"Results saved to: {results_path}")

        return metrics, best_params

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    # List of manually selected files (can be namefile-str or DatasetType)
    SELECTED_CSV = [
        DatasetType.MY_AUDIO
    ]

    # Filtering CSV files by feature type and filters
    csv_paths = find_csv_paths(
        feature_type="mfcc",
        bandpass_filter_type="butter",
        denoising_filter_type="savgol"
    )

    selected_filenames = [item.value if isinstance(item, DatasetType) else item for item in SELECTED_CSV]
    filtered_filenames = [os.path.basename(path) for path in csv_paths]
    all_filenames = list(set(selected_filenames) | set(filtered_filenames))

    model_types = ModelType.get_all_models()
    # model_types = [
    #     ModelType.CATBOOST,
    #     ModelType.RANDOM_FOREST,
    #     # ModelType.VOTING,
    #     # ModelType.STACKING,
    #     # ModelType.QDA
    # ]

    for dataset_filename in all_filenames:
        logging.info(f"Training on file: {DatasetType.get_dataset_path(dataset_filename)}")
        for model_type in model_types:
            try:
                train_and_evaluate_model(model_type, dataset_filename)
            except Exception as e:
                logging.info(f"Error - {dataset_filename} and {model_type}: {e}")
