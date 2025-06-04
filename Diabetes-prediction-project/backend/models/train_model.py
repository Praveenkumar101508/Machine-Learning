import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from .diabetics_model import create_models
from config import OUTPUT_GRAPHS, RANDOM_STATE  # Import OUTPUT_GRAPHS directory path

# Set the backend for Matplotlib to Agg to ensure it works without a display
matplotlib.use('Agg')

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_GRAPHS, f'confusion_matrix_{model_name}.png'))
    plt.close()  # Ensure the plot is closed after saving

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(OUTPUT_GRAPHS, f'roc_curve_{model_name}.png'))
    plt.close()  # Ensure the plot is closed after saving

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    models = create_models()
    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Define parameter grid for each model
        if name == 'hist_gradient_boosting':
            param_grid = {
                'classifier__max_iter': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        elif name == 'logistic_regression':
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        elif name == 'random_forest':
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20]
            }
        elif name == 'decision_tree':
            param_grid = {
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10]
            }
        elif name == 'sgd_svm':
            param_grid = {
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__loss': ['hinge', 'modified_huber']
            }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Save confusion matrix and ROC curve
        plot_confusion_matrix(y_test, y_pred, name)
        plot_roc_curve(y_test, y_pred_proba, name)

        results[name] = {
            'model': best_model,
            'metrics': metrics
        }

    return results
