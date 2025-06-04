import os
import tempfile
import traceback
import uuid
import hashlib
import re
from typing import Dict, Any, List
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import joblib
import pandas as pd
import PyPDF2
from services.prediction_service import predict_diabetes
from services.data_service import add_engineered_features, load_and_preprocess_data
from services.model_service import train_and_evaluate_models, save_model
from config import MODEL_PATH, VARIABLES_PATH, PDF_DIR, OUTPUT_GRAPHS
from models.diabetics_model import CustomHistGradientBoostingClassifier # type: ignore
from services.utils import extract_features_from_pdf
from utils.helpers import print_colored, allowed_file, create_output_dir, save_plot, load_variables

app = Flask(__name__)
CORS(app)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and preprocessor
best_model = None
preprocessor = None

def load_or_train_model():
    global best_model, preprocessor
    if os.path.exists(MODEL_PATH):
        best_model, preprocessor = joblib.load(MODEL_PATH)
        print_colored("Model loaded from file.", "green")
    else:
        print_colored("Model not found. Training a new model...", "yellow")
        X_train, X_test, y_train, y_test, preprocessor, data = load_and_preprocess_data()
        results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        best_model_name = max(results, key=lambda x: results[x]['test_metrics']['roc_auc'])
        best_model = results[best_model_name]['model']
        save_model(best_model, preprocessor)

# Load or train the model when the app starts
load_or_train_model()

def get_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def generate_reason(features: Dict[str, Any], prediction: float) -> str:
    reason = f"Based on the analysis, there's a {'high' if prediction >= 0.5 else 'low'} likelihood of diabetes (probability: {prediction:.4f}). "
    risk_factors = []
    
    if features.get('HighBP', 0) == 1:
        risk_factors.append("High blood pressure")
    if features.get('HighChol', 0) >= 200:
        risk_factors.append("High cholesterol")
    if features.get('BMI', 0) >= 30:
        risk_factors.append(f"High BMI ({features['BMI']:.1f})")
    if features.get('Smoker', 0) == 1:
        risk_factors.append("Smoking")
    if features.get('PhysActivity', 0) == 0:
        risk_factors.append("Lack of physical activity")
    if features.get('HvyAlcoholConsump', 0) == 1:
        risk_factors.append("Heavy alcohol consumption")
    if features.get('GenHlth', 0) >= 4:
        risk_factors.append("Poor general health")
    
    if risk_factors:
        reason += f"Key contributing factors include: {', '.join(risk_factors)}. "
    else:
        reason += "No significant risk factors were identified."
    
    return reason

def generate_suggestions(status: str) -> List[str]:
    suggestions = []
    if status == "At Risk":
        suggestions = [
            "Consult with a healthcare provider for further evaluation and diagnosis.",
            "Consider lifestyle changes such as a balanced diet, regular exercise, and weight management.",
            "Monitor blood sugar levels regularly.",
            "Reduce stress and ensure adequate sleep.",
            "Explore diabetic management plans, if confirmed."
        ]
    else:
        suggestions = [
            "Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.",
            "Keep up with routine health check-ups to monitor key health indicators.",
            "Stay informed about diabetes prevention strategies.",
            "Maintain a healthy weight and avoid smoking.",
            "Ensure a diet rich in fruits, vegetables, and whole grains."
        ]
    return suggestions

@app.route('/verify_file', methods=['POST'])
def verify_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)
        file_hash = get_file_hash(temp_path)
        file_size = os.path.getsize(temp_path)
        os.remove(temp_path)
        return jsonify({"filename": file.filename, "hash": file_hash, "size": file_size})

@app.route('/predict', methods=['POST'])
def predict():
    request_id = str(uuid.uuid4())
    logger.info(f"Request ID: {request_id}, Starting prediction process")
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            logger.info(f"Request ID: {request_id}, File saved temporarily: {temp_path}")
            
            file_hash = get_file_hash(temp_path)
            file_size = os.path.getsize(temp_path)
            logger.info(f"Request ID: {request_id}, File hash: {file_hash}, File size: {file_size}")

            features = extract_features_from_pdf(temp_path)
            logger.info(f"Request ID: {request_id}, Extracted features: {features}")

            df = pd.DataFrame([features])
            df = add_engineered_features(df)
            logger.info(f"Request ID: {request_id}, DataFrame after engineering: {df}")

            preprocessed = preprocessor.transform(df)
            logger.info(f"Request ID: {request_id}, Preprocessed features: {preprocessed}")

            prediction = predict_diabetes(features, best_model, preprocessor)
            logger.info(f"Request ID: {request_id}, Prediction: {prediction}")

            status = prediction['PredictedStatus']
            reason = generate_reason(features, prediction['DiabetesProbability'])
            suggestions = generate_suggestions(status)

            response = {
                "request_id": request_id,
                "PredictedStatus": status,
                "DiabetesProbability": prediction['DiabetesProbability'],
                "Reason": reason,
                "Suggestions": suggestions
            }

            os.unlink(temp_path)

            return jsonify(response)
        except Exception as e:
            logger.error(f"Request ID: {request_id}, Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

@app.route('/distribution_data')
def get_distribution_data():
    _, data = joblib.load(VARIABLES_PATH)
    distribution = data['Diabetes_binary'].value_counts().reset_index()
    distribution.columns = ['name', 'value']
    distribution['name'] = distribution['name'].map({0: 'No Diabetes', 1: 'Diabetes'})
    return jsonify(distribution.to_dict('records'))

@app.route('/feature_importance')
def get_feature_importance():
    if not hasattr(best_model, 'feature_importances_'):
        return jsonify({"error": "Feature importance not available for this model"}), 400
    _, data = joblib.load(VARIABLES_PATH)
    feature_importance = pd.DataFrame({
        'name': data.drop('Diabetes_binary', axis=1).columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    return jsonify(feature_importance.to_dict('records'))

@app.route('/correlation_data')
def get_correlation_data():
    _, data = joblib.load(VARIABLES_PATH)
    correlation_matrix = data.corr()
    correlation_data = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            correlation_data.append({
                'x': correlation_matrix.columns[i],
                'y': correlation_matrix.columns[j],
                'value': abs(correlation_matrix.iloc[i, j]) * 1000  # Scale up for better visibility
            })
    return jsonify(correlation_data)

if __name__ == '__main__':
    app.run(debug=True)
