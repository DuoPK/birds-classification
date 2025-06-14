def get_default_params(model_name, input_size=None, output_size=None):
    """Get default parameters for the model"""
    if model_name == 'NeuralNetworkModel':
        return {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_sizes': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'dropout_rate': 0.2
        }
    elif model_name == 'SVCModel':
        return {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    elif model_name == 'CatBoostModel':
        return {'iterations': 100, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3}
    elif model_name == 'XGBoostModel':
        return {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1}
    elif model_name == 'RandomForestModel':
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'criterion': 'gini'
        }
    elif model_name == 'LogisticRegressionModel':
        return {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs', 'penalty': 'l2'}
    elif model_name == 'GaussianNBModel':
        return {'var_smoothing': 1e-9}
    elif model_name == 'QuadraticDiscriminantAnalysisModel':
        return {'reg_param': 0.0}
    elif model_name == 'VotingModel':
        return {
            'voting': 'soft',
            'weights': [1.0, 1.0, 1.0],
            'lr_params': {'C': 1.0, 'max_iter': 1000},
            'gnb_params': {'var_smoothing': 1e-9},
            'qda_params': {'reg_param': 0.0}
        }
    elif model_name == 'StackingModel':
        return {
            'cv': 5,
            'lr_params': {'C': 1.0, 'max_iter': 1000},
            'gnb_params': {'var_smoothing': 1e-9},
            'qda_params': {'reg_param': 0.0},
            'final_estimator_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        }
    else:
        raise ValueError(f"Unknown model: {model_name}") 