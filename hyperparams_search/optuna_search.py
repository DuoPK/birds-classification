from datetime import datetime

import optuna
import numpy as np
from optuna import TrialPruned
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
import time
import logging
from utils.ResultsLogger import ResultsLogger
from utils.ClassificationMetrics import ClassificationMetrics
from utils.CrossValidator import CrossValidator
from utils.config import (
    CV, N_TRIALS, RANDOM_STATE, SCORING_OPTUNA,
    USE_OPTUNA_FEATURE_SELECTION, MIN_FEATURES_TO_SELECT, MAX_FEATURES_TO_SELECT
)
from hyperparams_search.default_params import get_default_params

optuna.logging.set_verbosity(optuna.logging.DEBUG)


class OptunaSearch:
    def __init__(self, model_class, model_name, input_size=None, output_size=None,
                 n_trials=N_TRIALS, cv=CV, scoring=SCORING_OPTUNA, random_state=RANDOM_STATE,
                 log_default_params=True, use_select_kbest=False, k_best=None, feature_names=None,
                 time_score_weight=None):
        """
        Parameters:
        - model_class: class of the model to optimize
        - model_name: name of the model (used to get parameter space)
        - input_size: input size for neural network
        - output_size: output size for neural network
        - n_trials: number of optimization trials
        - cv: number of cross-validation folds
        - scoring: metric to optimize ('accuracy' or 'f1_score')
        - random_state: random seed for reproducibility
        - use_select_kbest: whether to use SelectKBest for feature selection
        - k_best: number of features to select if use_select_kbest is True
        - feature_names: list of feature names for Optuna feature selection
        """
        self.model_class = model_class
        self.model_name = model_name
        self.input_size = input_size
        self.output_size = output_size
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.trials_ = []
        self.log_default_params = log_default_params
        self.use_select_kbest = use_select_kbest
        self.k_best = k_best
        self.feature_names = feature_names
        self.time_score_weight = time_score_weight
        
        results_base_dir_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.log_default_params:
            self.results_logger = ResultsLogger(results_base_dir_date)
        else:
            self.results_logger = None
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        
    def _objective(self, trial, X, y):
        """Objective function for Optuna optimization using sklearn CV"""
        model_params = None
        feature_params = None
        try:
            # Get parameter suggestions based on the model type
            if self.model_name == 'NeuralNetworkModel':
                model_params = self._suggest_neural_network_params(trial)
                # Log neural network structure for each trial
                model = self.model_class(**model_params)
                self.logger.info(f"\nTrial {len(self.trials_) + 1} - {model.get_model_structure()}")
            elif self.model_name == 'SVCModel':
                model_params = self._suggest_svm_params(trial)
            elif self.model_name == 'CatBoostModel':
                model_params = self._suggest_catboost_params(trial)
            elif self.model_name == 'XGBoostModel':
                model_params = self._suggest_xgboost_params(trial)
            elif self.model_name == 'RandomForestModel':
                model_params = self._suggest_random_forest_params(trial)
            elif self.model_name == 'LogisticRegressionModel':
                model_params = self._suggest_logistic_regression_params(trial)
            elif self.model_name == 'GaussianNBModel':
                model_params = self._suggest_gaussian_nb_params(trial)
            elif self.model_name == 'QuadraticDiscriminantAnalysisModel':
                model_params = self._suggest_qda_params(trial)
            elif self.model_name == 'VotingModel':
                model_params = self._suggest_voting_params(trial)
            elif self.model_name == 'StackingModel':
                model_params = self._suggest_stacking_params(trial)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Add feature selection parameters if enabled
            if USE_OPTUNA_FEATURE_SELECTION and self.feature_names is not None:
                feature_params = self._suggest_feature_selection_params(trial)
            
            # Sklearn Cross-validation for optimization
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply feature selection if enabled
                if USE_OPTUNA_FEATURE_SELECTION and self.feature_names is not None:
                    X_train, X_val = self._apply_feature_selection(X_train, X_val, feature_params)
                
                # Initialize and train the model
                model = self.model_class(**model_params)
                if hasattr(model, 'train'):
                    model.train(np.array(X_train, copy=True), np.array(y_train, copy=True))
                
                y_pred = model.predict(X_val)
                
                # Calculate score using ClassificationMetrics
                metrics = ClassificationMetrics(y_val, y_pred)
                if self.scoring == 'accuracy':
                    score = metrics.accuracy()
                elif self.scoring == 'f1_score':
                    score = metrics.f1_score()
                else:
                    raise ValueError(f"Unknown scoring metric: {self.scoring}")
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Store trial results
            trial_params = model_params.copy()
            if feature_params is not None:
                trial_params.update(feature_params)
                
            self.trials_.append({
                'params': trial_params,
                'mean_score': mean_score,
                'std_score': std_score,
                'status': 'success'
            })

            # Add to the score number of selected features if feature selection is used
            # The fewer selected_features_count, the better; the higher the score the better
            if USE_OPTUNA_FEATURE_SELECTION and self.feature_names is not None and self.time_score_weight is not None:
                selected_features_count = sum(1 for v in feature_params.values() if v == 1)
                mean_score -= selected_features_count * self.time_score_weight
            return mean_score
            
        except Exception as e:
            # Log the error
            self.logger.error(f"Trial {len(self.trials_) + 1} failed with parameters: {model_params} because of the following error: {str(e)}")
            
            # Store failed trial results
            trial_params = model_params.copy() if model_params is not None else {}
            if feature_params is not None:
                trial_params.update(feature_params)
                
            self.trials_.append({
                'params': trial_params,
                'error': str(e),
                'status': 'failed'
            })
            
            # Return a very low score to indicate failure
            return float('-inf')
    
    def _suggest_neural_network_params(self, trial):
        # First suggest the number of layers
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        # Then suggest the size for each layer
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'layer_{i}_size', 32, 256, step=32)
            hidden_sizes.append(size)
        
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': hidden_sizes,
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 50, 200, step=50),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.1)
        }
    
    def _suggest_svm_params(self, trial):
        return {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.1, 0.01])
        }
    
    def _suggest_catboost_params(self, trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8, step=2),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5, step=2)
        }
    
    def _suggest_xgboost_params(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8, step=1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5, step=2)
        }
    
    def _suggest_random_forest_params(self, trial):
        """Suggest hyperparameters for Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 16, step=2),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=2),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
    
    def _suggest_logistic_regression_params(self, trial):
        """Suggest hyperparameters for LogisticRegression with static search space and logical validation."""

        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 2000, step=100)

        # Map of correct solver-penalty combinations
        valid_combinations = {
            'lbfgs':      ['l2', None],
            'newton-cg':  ['l2', None],
            'liblinear':  ['l1', 'l2'],
            'sag':        ['l2', None],
            'saga':       ['l1', 'l2', 'elasticnet', None],
        }

        if penalty not in valid_combinations.get(solver, []):
            raise TrialPruned()  # reject an invalid combination

        # For elasticnet l1_ratio is required
        params = {
            'solver': solver,
            'penalty': penalty,
            'class_weight': class_weight,
            'C': C,
            'max_iter': max_iter,
        }

        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        return params
    
    def _suggest_gaussian_nb_params(self, trial):
        """Suggest hyperparameters for Gaussian Naive Bayes"""
        return {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-9, 1e-3, log=True)
        }
    
    def _suggest_qda_params(self, trial):
        """Suggest hyperparameters for Quadratic Discriminant Analysis"""
        return {
            'reg_param': trial.suggest_float('reg_param', 0.0, 1.0)
        }
    
    def _suggest_voting_params(self, trial):
        """Suggest hyperparameters for Voting Classifier"""
        # Suggest parameters for base models
        lr_params = {
            'C': trial.suggest_float('lr_C', 0.001, 100.0, log=True),
            'max_iter': trial.suggest_int('lr_max_iter', 100, 2000, step=100)
        }
        gnb_params = {
            'var_smoothing': trial.suggest_float('gnb_var_smoothing', 1e-9, 1e-3, log=True)
        }
        qda_params = {
            'reg_param': trial.suggest_float('qda_reg_param', 0.0, 1.0)
        }

        # Suggest weights for each base model
        weights = [
            trial.suggest_float('voting_weight_lr', 0.1, 2.0),
            trial.suggest_float('voting_weight_gnb', 0.1, 2.0),
            trial.suggest_float('voting_weight_qda', 0.1, 2.0)
        ]
        
        return {
            'voting': trial.suggest_categorical('voting', ['hard', 'soft']),
            'weights': weights,
            'lr_params': lr_params,
            'gnb_params': gnb_params,
            'qda_params': qda_params
        }
    
    def _suggest_stacking_params(self, trial):
        """Suggest hyperparameters for Stacking Classifier"""
        # Suggest parameters for base models
        lr_params = {
            'C': trial.suggest_float('lr_C', 0.001, 100.0, log=True),
            'max_iter': trial.suggest_int('lr_max_iter', 100, 2000, step=100)
        }
        gnb_params = {
            'var_smoothing': trial.suggest_float('gnb_var_smoothing', 1e-9, 1e-3, log=True)
        }
        qda_params = {
            'reg_param': trial.suggest_float('qda_reg_param', 0.0, 1.0)
        }
        
        # Suggest parameters for final estimator (Random Forest)
        final_estimator_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200, step=50),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4)
        }
        
        return {
            'cv': trial.suggest_int('cv', 3, 7),
            'lr_params': lr_params,
            'gnb_params': gnb_params,
            'qda_params': qda_params,
            'final_estimator_params': final_estimator_params
        }
    
    def _get_default_params(self):
        """Get default parameters for the model"""
        return get_default_params(self.model_name, self.input_size, self.output_size)

    def _evaluate_default_params(self, X, y, dataset_name):
        """Evaluate the model with default parameters using both custom and sklearn CV"""
        default_params = self._get_default_params()
        model = self.model_class(**default_params)
        
        # Custom CV evaluation
        custom_cv_start = time.time()
        custom_cv = CrossValidator(model, k=self.cv, random_state=self.random_state)
        custom_cv_results = custom_cv.evaluate(X, y)
        custom_cv_time = time.time() - custom_cv_start
        custom_cv_results['training_time'] = custom_cv_time
        
        # Sklearn CV evaluation for comparison
        sklearn_cv_start = time.time()
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        sk_acc_scores = []
        sk_f1_scores = []
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            model = self.model_class(**default_params)
            if hasattr(model, 'train'):
                model.train(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            y_pred_proba = model.predict_proba(X_fold_val)
            
            # Calculate metrics using ClassificationMetrics
            metrics = ClassificationMetrics(y_fold_val, y_pred)
            sk_acc_scores.append(metrics.accuracy())
            sk_f1_scores.append(metrics.f1_score())
            
            y_true_all.extend(y_fold_val)
            y_pred_all.extend(y_pred)
            y_pred_proba_all.extend(y_pred_proba)
        
        sklearn_cv_time = time.time() - sklearn_cv_start
        sklearn_cv_results = {
            'mean_accuracy': np.mean(sk_acc_scores),
            'std_accuracy': np.std(sk_acc_scores),
            'mean_f1_score': np.mean(sk_f1_scores),
            'std_f1_score': np.std(sk_f1_scores),
            'training_time': sklearn_cv_time
        }

        if self.log_default_params:
            # Log results
            self.results_logger.log_default_params_results(
                dataset_name, self.model_name, custom_cv_results, sklearn_cv_results
            )
            # Save plots
            self.results_logger.save_confusion_matrix(
                np.array(y_true_all), np.array(y_pred_all), dataset_name, self.model_name
            )
            self.results_logger.save_roc_curve(
                np.array(y_true_all), np.array(y_pred_proba_all), dataset_name, self.model_name
            )
        
        return default_params

    def _suggest_feature_selection_params(self, trial):
        """Suggest feature selection parameters for Optuna"""
        feature_params = {}
        
        # Suggest binary parameters for all features in feature_names
        for feature in self.feature_names:
            feature_params[f'feature_{feature}'] = trial.suggest_categorical(f'feature_{feature}', [0, 1])
        
        return feature_params
    
    def _apply_feature_selection(self, X_train, X_val, feature_params):
        """Apply feature selection based on Optuna parameters"""
        # Get selected feature indices
        selected_indices = []
        for i, feature in enumerate(self.feature_names):
            param_name = f'feature_{feature}'
            if param_name in feature_params and feature_params[param_name] == 1:
                selected_indices.append(i)
        
        # Ensure a minimum number of features
        if len(selected_indices) < MIN_FEATURES_TO_SELECT:
            # If too few features selected, add some randomly
            remaining_indices = [i for i in range(len(self.feature_names)) if i not in selected_indices]
            additional_indices = np.random.choice(
                remaining_indices,
                size=MIN_FEATURES_TO_SELECT - len(selected_indices),
                replace=False
            )
            selected_indices.extend(additional_indices)
        
        # Apply selection
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        
        return X_train_selected, X_val_selected

    def fit(self, X, y, dataset_name):
        """Find the best parameters using Optuna optimization"""
        # Create study and enqueue default parameters trial

        storage_name = "sqlite:///alzheimer_classification_v3.db"
        # Add k_best to the storage name if specified
        if self.use_select_kbest and self.k_best is not None:
            storage_name = storage_name.replace('.db', f'_select-{self.k_best}-best.db')
        # Add feature selection to storage name if enabled
        if USE_OPTUNA_FEATURE_SELECTION:
            storage_name = storage_name.replace('.db', '_optuna-feature-selection.db')
            
        study = optuna.create_study(direction='maximize',
                                    storage=storage_name,
                                    load_if_exists=True,
                                    study_name=f"{self.model_name}_{dataset_name}",
                                    sampler=TPESampler(seed=self.random_state))

        # First, add run with default parameters
        if self.log_default_params:
            # Evaluate and log - using both custom and sklearn CV
            default_params = self._evaluate_default_params(X, y, dataset_name)
        else:
            default_params = self._get_default_params()
            
        # Add feature selection parameters to default params if enabled
        if USE_OPTUNA_FEATURE_SELECTION and self.feature_names is not None:
            for feature in self.feature_names:
                default_params[f'feature_{feature}'] = 1
                
        self.logger.info("Add default parameters before optimization...")
        study.enqueue_trial(default_params)
        
        # Run optimization using sklearn CV
        study.optimize(lambda trial: self._objective(trial, X, y), 
                      n_trials=self.n_trials)
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # Log results
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self
    