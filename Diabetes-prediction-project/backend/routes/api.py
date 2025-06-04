from flask import Blueprint, request, jsonify, send_from_directory, current_app as app
from services.prediction_service import predict_diabetes
from services.data_service import extract_features_from_pdf, load_and_preprocess_data
from services.model_service import load_model_and_preprocessor
from services.eda_service import perform_eda
from utils.helpers import allowed_file
import pandas as pd
import numpy as np
import joblib
import traceback
import os
import logging
from werkzeug.utils import secure_filename
from config import PDF_DIR, VARIABLES_PATH, DATA_PATH

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

@api_bp.route('/', defaults={'path': ''})
@api_bp.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(PDF_DIR, filename)
            file.save(filepath)
            extracted_features = extract_features_from_pdf(filepath)
            if extracted_features:
                pdf_data = pd.Series(extracted_features)
                best_model, preprocessor = load_model_and_preprocessor()
                prediction = predict_diabetes(pdf_data, best_model, preprocessor)
                os.remove(filepath)  # Remove the file after processing
                return jsonify(prediction)
            else:
                return jsonify({"error": "Failed to extract features from PDF"}), 400
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

@api_bp.route('/distribution_data', methods=['GET'])
def get_distribution_data():
    try:
        _, data = joblib.load(VARIABLES_PATH)
        if data is None:
            return jsonify({"error": "Data not loaded"}), 500
        
        distribution = data['Diabetes_binary'].value_counts().reset_index()
        distribution.columns = ['name', 'value']
        distribution['name'] = distribution['name'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        return jsonify(distribution.to_dict('records'))
    except Exception as e:
        logger.error(f"Error in get_distribution_data: {str(e)}")
        return jsonify({"error": "An error occurred while fetching distribution data"}), 500

@api_bp.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    try:
        best_model, data = joblib.load(VARIABLES_PATH)
        
        if best_model is None:
            return jsonify({"error": "Model not trained"}), 500
        
        feature_importance = []
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            feature_importance = abs(best_model.coef_[0])
        else:
            return jsonify({"error": "Feature importance not available for this model"}), 400
        
        if data is None:
            return jsonify({"error": "Data not loaded"}), 500
        
        feature_importance_data = pd.DataFrame({
            'name': data.drop('Diabetes_binary', axis=1).columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return jsonify(feature_importance_data.to_dict('records'))
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {str(e)}")
        return jsonify({"error": "An error occurred while fetching feature importance"}), 500

@api_bp.route('/correlation_data', methods=['GET'])
def get_correlation_data():
    try:
        _, data = joblib.load(VARIABLES_PATH)
        
        if data is None:
            return jsonify({"error": "Data not loaded"}), 500
        
        correlation_matrix = data.drop('Diabetes_binary', axis=1).corr()
        correlation_data = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                correlation_data.append({
                    'x': correlation_matrix.columns[i],
                    'y': correlation_matrix.columns[j],
                    'value': correlation_matrix.iloc[i, j]
                })
        
        return jsonify(correlation_data)
    except Exception as e:
        logger.error(f"Error in get_correlation_data: {str(e)}")
        return jsonify({"error": "An error occurred while fetching correlation data"}), 500

@api_bp.route('/eda', methods=['GET'])
def get_eda():
    try:
        df, _, _, _, _, _ = load_and_preprocess_data(DATA_PATH)
        eda_results = perform_eda(df)
        return jsonify(eda_results)
    except Exception as e:
        logger.error(f"Error in get_eda: {str(e)}")
        return jsonify({"error": "An error occurred while performing EDA"}), 500

@api_bp.errorhandler(500)
def internal_server_error(error):
    return jsonify(error=str(error), stacktrace=traceback.format_exc()), 500