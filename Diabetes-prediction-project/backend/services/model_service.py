from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from config import CLASSIFICATION_THRESHOLD, MODEL_PATH, RANDOM_STATE, OUTPUT_GRAPHS
import joblib
import numpy as np
from typing import Dict, Any
from sklearn.calibration import CalibratedClassifierCV
from models.diabetics_model import create_models
from utils.helpers import print_colored
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def train_and_evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
    models = create_models()
    results = {}
    for name, model in models.items():
        print_colored(f"\nTraining and evaluating {name}...", "cyan")
        model.fit(X_train, y_train)

        # Check if the model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_train_pred = (model.predict_proba(X_train)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test_pred = (model.predict_proba(X_test)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
            y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Use decision_function for models that don't support predict_proba
            y_train_pred = (model.decision_function(X_train) >= 0).astype(int)
            y_test_pred = (model.decision_function(X_test) >= 0).astype(int)
            y_test_pred_proba = model.decision_function(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_test_pred)
        cr = classification_report(y_test, y_test_pred)

        # Save confusion matrix and ROC curve
        save_confusion_matrix(y_test, y_test_pred, name)
        save_roc_curve(y_test, y_test_pred_proba, name)

        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': cm,
            'classification_report': cr,
            'test_predictions': y_test_pred,  # Added this to avoid the KeyError
        }

    # Plot model comparison
    plot_model_comparison(results)

    return results

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    return metrics

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f"{OUTPUT_GRAPHS}/{model_name}_confusion_matrix.png")
    plt.close()
    print_colored(f"\nConfusion matrix saved for {model_name}", "green")

def save_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f"{OUTPUT_GRAPHS}/{model_name}_roc_curve.png")
    plt.close()
    print_colored(f"\nROC curve saved for {model_name}", "green")

def print_metrics(title, metrics):
    print_colored(title, "green")
    for metric, value in metrics.items():
        print_colored(f"{metric.capitalize()}: {value:.4f}", "yellow")

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print_colored("\nConfusion Matrix:", "green")
    print(cm)

def print_classification_report(y_true, y_pred, y_pred_proba):
    print_colored("\nClassification Report:", "green")
    print(classification_report(y_true, y_pred))
    print_colored(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}", "yellow")

def save_model(model, preprocessor):
    joblib.dump((model, preprocessor), MODEL_PATH)
    print_colored(f"\nModel and preprocessor saved as '{MODEL_PATH}'", "green")

def load_model_and_preprocessor():
    return joblib.load(MODEL_PATH)

def plot_model_comparison(results: Dict[str, Dict]):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        rects = plt.bar(x + offset, [results[model]['test_metrics'][metric] for model in model_names], width, label=metric)
        multiplier += 1
    
    plt.ylabel('Scores')
    plt.xlabel('Models')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, model_names, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_GRAPHS}/model_comparison.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_GRAPHS}/feature_importance.png")
        plt.close()
    else:
        print_colored("This model doesn't have feature_importances_ attribute.", "yellow")
