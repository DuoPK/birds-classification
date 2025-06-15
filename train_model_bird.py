import os
import pandas as pd
import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import joblib
import seaborn as sns
from utils.enums import ModelType, DatasetType
from utils.ResultsLogger import ResultsLogger
from models.RandomForestModel import RandomForestModel
from models.SVCModel import SVCModel
from models.CatBoostModel import CatBoostModel
from models.XGBoostModel import XGBoostModel
from models.LogisticRegressionModel import LogisticRegressionModel
from models.GaussianNBModel import GaussianNBModel
from models.QuadraticDiscriminantAnalysisModel import QuadraticDiscriminantAnalysisModel
from models.VotingModel import VotingModel
from models.StackingModel import StackingModel
from sklearn.feature_selection import SelectKBest, f_classif

# === PARAMETRY ===
RANDOM_STATE = 42
N_JOBS = -1
CV = 5
N_TRIALS = 10
OPTUNA_TIMEOUT = 300  # 5 minutes
SELECT_KBEST = True
K_BEST = 10

MODELS_DIR = "models/saved"
RESULTS_DIR = "results"

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_dataset(dataset_type: DatasetType):
    """Load and validate dataset"""
    filepath = dataset_type.get_dataset_path(dataset_type)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if 'Diagnosis' not in df.columns or 'SourceFile' not in df.columns:
        raise ValueError("Dataset must contain 'Diagnosis' and 'SourceFile' columns")
    
    # Drop metadata columns
    metadata_cols = ['Diagnosis', 'Segment', 'SourceFile']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    return df, feature_cols

def split_data_by_source(df, feature_cols, test_size=0.2, random_state=RANDOM_STATE):
    """Split data based on source files to prevent data leakage"""
    # Get unique source files
    source_files = df['SourceFile'].unique()
    if len(source_files) < 2:
        raise ValueError("Not enough unique source files for splitting")
    
    # Split source files into train and test
    train_files, test_files = train_test_split(
        source_files,
        test_size=test_size,
        random_state=random_state,
        stratify=df.groupby('SourceFile')['Diagnosis'].first()
    )
    
    # Split data based on source files
    train_data = df[df['SourceFile'].isin(train_files)]
    test_data = df[df['SourceFile'].isin(test_files)]
    
    if train_data.empty or test_data.empty:
        raise ValueError("Empty dataset after splitting")
    
    # Prepare feature matrices and labels
    X_train = train_data[feature_cols]
    y_train = train_data['Diagnosis']
    X_test = test_data[feature_cols]
    y_test = test_data['Diagnosis']
    
    if X_train.empty or X_test.empty:
        raise ValueError("Empty feature matrix after splitting")
    
    return X_train, X_test, y_train, y_test, train_files, test_files

def train_and_evaluate_model(model_type: ModelType, dataset_type: DatasetType, use_select_kbest=SELECT_KBEST, k_best=K_BEST):
    """Train and evaluate model using scikit-learn's cross-validation"""
    try:
        logging.info(f"Starting training for {model_type.value} on {dataset_type.value}")
        
        # Load dataset
        df, feature_cols = load_dataset(dataset_type)
        logging.info(f"Loaded dataset with {len(df)} samples and {len(feature_cols)} features")
        
        # Split data
        X_train, X_test, y_train, y_test, train_files, test_files = split_data_by_source(
            df, feature_cols, test_size=0.2, random_state=RANDOM_STATE
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

        study_name = f'{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_kbest-{k_best}' if use_select_kbest else f'{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}'
        # Perform hyperparameter optimization
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            load_if_exists=True,
            storage='sqlite:///optuna_study_sound.db',
        )
        
        def objective(trial):
            params = model.get_optuna_params(trial)
            model_instance = model.get_model_instance(params)
            scores = cross_val_score(
                model_instance, X_train, y_train_encoded,
                cv=CV, scoring='f1_weighted', n_jobs=N_JOBS
            )
            return scores.mean()
        
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)
        
        # Train final model with best parameters
        best_params = model.get_params_from_optuna_params(study.best_params)
        final_model = model.get_model_instance(best_params)
        final_model.fit(X_train, y_train_encoded)
        
        # Evaluate on test set
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
        
        # Save results
        logger = ResultsLogger(dataset_type.get_dataset_name(dataset_type))
        results = {
            'model_type': model_type.value,
            'dataset': dataset_type.value,
            'best_params': best_params,
            'metrics': metrics,
            'study_name': study.study_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metrics to CSV
        os.makedirs(RESULTS_DIR, exist_ok=True)
        reslts_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_results.csv"
        if use_select_kbest:
            reslts_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_kbest-{k_best}_results.csv"
        results_path = os.path.join(RESULTS_DIR, reslts_file)
        
        # Convert metrics to flat dictionary for CSV
        flat_metrics = {
            'model_type': model_type.value,
            'dataset': dataset_type.value,
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
        cm_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_confusion_matrix.csv"
        if use_select_kbest:
            cm_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_kbest-{k_best}_confusion_matrix.csv"
        cm_path = os.path.join(RESULTS_DIR, cm_path_file)
        pd.DataFrame(metrics['confusion_matrix']).to_csv(cm_path, index=False)

        class_labels = label_encoder.classes_
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"Confusion Matrix: {model_type.value} na {dataset_type.get_dataset_name(dataset_type)}")
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywista")
        img_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_confusion_matrix.png"
        if use_select_kbest:
            img_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_kbest-{k_best}_confusion_matrix.png"
        img_path = os.path.join(RESULTS_DIR, img_path_file)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        logging.info(f"Obrazek macierzy pomyłek zapisany: {img_path}")

        # Save model and results
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}.joblib"
        if use_select_kbest:
            model_path_file = f"{model_type.value}_{dataset_type.get_dataset_name(dataset_type)}_kbest-{k_best}.joblib"
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

def get_model_instance(model_type: ModelType):
    """Get model instance based on model type"""
    model_classes = {
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.SVM: SVCModel,
        ModelType.CATBOOST: CatBoostModel,
        ModelType.XGBOOST: XGBoostModel,
        ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
        ModelType.GAUSSIAN_NB: GaussianNBModel,
        ModelType.QDA: QuadraticDiscriminantAnalysisModel,
        ModelType.VOTING: VotingModel,
        ModelType.STACKING: StackingModel
    }
    return model_classes[model_type]()

if __name__ == "__main__":
    # Example usage
    model_type = ModelType.RANDOM_FOREST
    # model_type = ModelType.SVM
    # model_type = ModelType.CATBOOST
    # model_type = ModelType.XGBOOST
    dataset_type = DatasetType.MY_AUDIO
    
    try:
        results = train_and_evaluate_model(model_type, dataset_type)
        print("\nTraining completed successfully!")
        print(f"Best parameters: {results[0]}")
        print(f"Test metrics: {results[1]}")
    except Exception as e:
        print(f"\nError: {str(e)}")
