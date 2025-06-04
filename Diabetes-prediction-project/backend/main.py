import argparse
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple
from services.utils import extract_features_from_pdf
#from models.diabetes_model import CustomHistGradientBoostingClassifier # type: ignore
from utils.helpers import print_colored, create_output_dir, save_plot, load_variables
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.diabetics_model import CustomHistGradientBoostingClassifier

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, best_model, preprocessor  # Import the Flask app instance and model directly
from config import (DATA_PATH, MODEL_PATH, PDF_DIR, VARIABLES_PATH, OUTPUT_GRAPHS,
                    DISTRIBUTION_GRAPH, CORRELATION_HEATMAP, PAIRPLOT,
                    FEATURE_IMPORTANCE, MODEL_COMPARISON)
from services.data_service import load_and_preprocess_data
from services.model_service import train_and_evaluate_models, save_model, save_confusion_matrix, save_roc_curve
from services.prediction_service import predict_diabetes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_model_comparison(results: Dict[str, Dict]) -> None:
    """
    Generate and save a bar plot comparing model performances.
    
    Args:
        results (Dict[str, Dict]): Dictionary containing model results.
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    data = [
        {'Model': model, 'Metric': metric, 'Value': results[model]['test_metrics'][metric]}
        for model in models
        for metric in metrics
    ]
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    plt.title('Model Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON)
    print_colored(f"\nModel comparison graph saved as '{MODEL_COMPARISON}'", "green")
    plt.close()

def print_model_comparison_table(results: Dict[str, Dict]) -> None:
    """
    Print a table comparing model performances.
    
    Args:
        results (Dict[str, Dict]): Dictionary containing model results.
    """
    table = pd.DataFrame({
        model: result['test_metrics']
        for model, result in results.items()
    }).T
    
    print("\nModel Comparison Table:")
    print(table.to_string())

def test_sample_predictions(best_model, preprocessor: object) -> None:
    """
    Test sample predictions using PDF files.
    
    Args:
        best_model: The best performing model.
        preprocessor (object): Data preprocessor.
    """
    print("\nTesting sample predictions:")
    sample_pdfs = ['blood_sample_report_25.pdf', 'blood_sample_report_39.pdf']
    for pdf in sample_pdfs:
        pdf_path = os.path.join(PDF_DIR, pdf)
        features = extract_features_from_pdf(pdf_path)
        if features:
            prediction = predict_diabetes(features, best_model, preprocessor)
            print(f"\nPrediction for {pdf}:")
            print(prediction)
        else:
            print(f"Failed to extract features from {pdf}")

def run_training_pipeline(skip_eda: bool = True) -> Tuple[object, object, pd.DataFrame]:
    """
    Run the complete training pipeline.

    Args:
        skip_eda (bool): If True, skip the EDA steps.

    Returns:
        Tuple[object, object, pd.DataFrame]: Best model, preprocessor, and data.
    """
    print_colored("Loading and preprocessing data...", "cyan", "bright")
    X_train, X_test, y_train, y_test, preprocessor, data = load_and_preprocess_data(DATA_PATH, skip_eda=skip_eda)
    print(f"Dataset shape: {data.shape}")
    print(f"Training set shape after SMOTE and preprocessing: {X_train.shape}")
    print(f"Testing set shape after preprocessing: {X_test.shape}")

    if not skip_eda:
        print("Performing EDA...")
        # Additional EDA steps can be placed here if needed

    print_colored("\nStarting model training...", "cyan", "bright")
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    print_colored("Model training completed.", "green", "bright")

    best_model_name = None
    best_model = None
    best_roc_auc = 0.0

    for name, result in results.items():
        print_colored(f"\n{name}:", "green", "bright")
        print_colored("Training Metrics:", "yellow")
        for metric, value in result['train_metrics'].items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print_colored("\nTest Metrics:", "yellow")
        for metric, value in result['test_metrics'].items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print_colored("\nConfusion Matrix:", "yellow")
        print(result['confusion_matrix'])
        print_colored("\nClassification Report:", "yellow")
        print(result['classification_report'])

        # Check for the best model based on ROC AUC
        if result['test_metrics']['roc_auc'] > best_roc_auc:
            best_roc_auc = result['test_metrics']['roc_auc']
            best_model_name = name
            best_model = result['model']

    if best_model:
        print_colored(f"\nBest model: {best_model_name}", "green", "bright")
        print_colored("Best model metrics:", "green")
        for metric, value in results[best_model_name]['test_metrics'].items():
            print_colored(f"{metric.capitalize()}: {value:.4f}", "yellow")

        print("Saving best model...")
        save_model(best_model, preprocessor)
        print_colored(f"\nModel and preprocessor saved as '{MODEL_PATH}'", "green", "bright")

        print("Saving variables for Flask routes...")
        joblib.dump((best_model, data), VARIABLES_PATH)
        print_colored(f"\nVariables saved for Flask routes as '{VARIABLES_PATH}'", "green", "bright")

        logger.info(f"Best model type: {type(best_model)}")
        logger.info(f"Best model feature_importances_ available: {hasattr(best_model, 'feature_importances_')}")
        logger.info(f"Data shape: {data.shape}")

        return best_model, preprocessor, data
    else:
        print_colored("No models were successfully trained.", "red", "bright")
        return None, None, None

def sample_predictions(best_model: object, preprocessor: object) -> None:
    """
    Generate sample predictions from PDF files.
    
    Args:
        best_model: The best performing model.
        preprocessor (object): Data preprocessor.
    """
    print_colored("\nSample Predictions from PDF files:", "cyan", "bright")
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    random.shuffle(pdf_files)

    positive_sample = None
    negative_sample = None

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        features = extract_features_from_pdf(pdf_path)
        if features:
            prediction = predict_diabetes(features, best_model, preprocessor)
            if prediction['PredictedStatus'] == 'At Risk' and not positive_sample:
                positive_sample = (pdf_file, prediction)
            elif prediction['PredictedStatus'] == 'Low Risk' and not negative_sample:
                negative_sample = (pdf_file, prediction)
        
        if positive_sample and negative_sample:
            break

    if positive_sample:
        print_colored(f"\nPositive Case ({positive_sample[0]}):", "yellow")
        print(positive_sample[1])
    else:
        print_colored("\nNo positive case found in the sample PDFs.", "yellow")

    if negative_sample:
        print_colored(f"\nNegative Case ({negative_sample[0]}):", "yellow")
        print(negative_sample[1])

def main() -> None:
    """Main function to run the Diabetic Detection backend."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Run Diabetic Detection backend")
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--perform-eda', action='store_true', help='Perform Exploratory Data Analysis')
    args = parser.parse_args()

    if not args.skip_training:
        best_model, preprocessor, _ = run_training_pipeline(skip_eda=not args.perform_eda)
        if best_model and preprocessor:
            test_sample_predictions(best_model, preprocessor)
            sample_predictions(best_model, preprocessor)
    else:
        print_colored("Skipping model training...", "yellow", "bright")
        # The model should already be loaded in the app
        if best_model is None or preprocessor is None:
            print_colored("Error: Model not loaded. Please run without --skip-training first.", "red", "bright")
            return

    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    print_colored("\nStarting Flask server...", "cyan", "bright")
    app.run(debug=True, port=5000, use_reloader=False)

if __name__ == '__main__':
    main()
