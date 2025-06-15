import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
from datetime import datetime
from utils.ClassificationMetrics import ClassificationMetrics


class ResultsLogger:
    def __init__(self, results_base_dir='results'):
        """Initialize ResultsLogger with directories for saving results"""
        self.results_dir = os.path.join('results', results_base_dir, 'metrics')
        self.plots_dir = os.path.join('results', results_base_dir, 'plots')
        self.metrics_file = os.path.join(self.results_dir, 'default_params_metrics.csv')

        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Define columns for metrics DataFrame
        self.columns = [
            'timestamp', 'dataset', 'model',
            'custom_cv_mean_accuracy', 'custom_cv_mean_f1_score',
            'custom_cv_std_accuracy', 'custom_cv_std_f1_score',
            'custom_cv_training_time',
            'sklearn_cv_mean_accuracy', 'sklearn_cv_mean_f1_score',
            'sklearn_cv_std_accuracy', 'sklearn_cv_std_f1_score',
            'sklearn_cv_training_time'
        ]

        # Load or create metrics DataFrame
        if os.path.exists(self.metrics_file):
            self.metrics_df = pd.read_csv(self.metrics_file)
            # Ensure all required columns exist
            for col in self.columns:
                if col not in self.metrics_df.columns:
                    self.metrics_df[col] = None
        else:
            self.metrics_df = pd.DataFrame(columns=self.columns)

        self.font_title_size = 20
        self.font_label_size = 18
        self.font_tick_size = 16
        self.font_annot_size = 20
        self.font_legend_size = 16

    def log_default_params_results(self, dataset_name, model_name, custom_cv_results, sklearn_cv_results):
        """Log results for default parameters"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_row = pd.DataFrame([{
            'timestamp': timestamp,
            'dataset': dataset_name,
            'model': model_name,
            'custom_cv_mean_accuracy': custom_cv_results.get('mean_accuracy'),
            'custom_cv_mean_f1_score': custom_cv_results.get('mean_f1_score'),
            'custom_cv_std_accuracy': custom_cv_results.get('std_accuracy'),
            'custom_cv_std_f1_score': custom_cv_results.get('std_f1_score'),
            'custom_cv_training_time': custom_cv_results.get('training_time'),
            'sklearn_cv_mean_accuracy': sklearn_cv_results.get('mean_accuracy'),
            'sklearn_cv_mean_f1_score': sklearn_cv_results.get('mean_f1_score'),
            'sklearn_cv_std_accuracy': sklearn_cv_results.get('std_accuracy'),
            'sklearn_cv_std_f1_score': sklearn_cv_results.get('std_f1_score'),
            'sklearn_cv_training_time': sklearn_cv_results.get('training_time')
        }])

        # Ensure all columns exist in new_row
        for col in self.columns:
            if col not in new_row.columns:
                new_row[col] = None

        # Concatenate with existing DataFrame
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        self.metrics_df.to_csv(self.metrics_file, index=False)

    def save_confusion_matrix(self, y_true, y_pred, dataset_name, model_name):
        metrics = ClassificationMetrics(y_true, y_pred)
        cm = metrics.confusion_matrix()
        labels = metrics.labels

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            annot_kws={'size': self.font_annot_size},
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}', fontsize=self.font_title_size)
        plt.ylabel('True Label', fontsize=self.font_label_size)
        plt.xlabel('Predicted Label', fontsize=self.font_label_size)
        plt.xticks(fontsize=self.font_tick_size)
        plt.yticks(fontsize=self.font_tick_size)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'confusion_matrix_{dataset_name}_{model_name}_{timestamp}.png'
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def save_roc_curve(self, y_true, y_pred_proba, dataset_name, model_name):
        # For multiclass, we'll create a one-vs-rest ROC curve for each class
        n_classes = y_pred_proba.shape[1]
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=self.font_label_size)
        plt.ylabel('True Positive Rate', fontsize=self.font_label_size)
        plt.title(f'ROC Curves - {model_name} on {dataset_name}', fontsize=self.font_title_size)
        plt.legend(loc="lower right", fontsize=self.font_legend_size)
        plt.xticks(fontsize=self.font_tick_size)
        plt.yticks(fontsize=self.font_tick_size)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'roc_curve_{dataset_name}_{model_name}_{timestamp}.png'
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
