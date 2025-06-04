import pandas as pd
from config import CLASSIFICATION_THRESHOLD
from typing import List, Dict, Any
from services.data_service import add_engineered_features

def get_risk_factors(blood_sample: Dict[str, Any]) -> List[str]:
    risk_factors = []
    if blood_sample.get('BMI', 0) > 30:
        risk_factors.append(f"High BMI ({blood_sample['BMI']:.1f})")
    if blood_sample.get('Age', 0) >= 45:
        risk_factors.append(f"Age ({blood_sample['Age']:.0f})")
    if blood_sample.get('HighBP', 0) == 1:
        risk_factors.append("High blood pressure")
    if blood_sample.get('HighChol', 0) == 1:
        risk_factors.append("High cholesterol")
    if blood_sample.get('Glucose', 0) >= 100:
        risk_factors.append(f"High glucose ({blood_sample['Glucose']:.1f})")
    if blood_sample.get('HbA1c', 0) >= 5.7:
        risk_factors.append(f"High HbA1c ({blood_sample['HbA1c']:.1f})")
    if blood_sample.get('Smoker', 0) == 1:
        risk_factors.append("Smoking")
    if blood_sample.get('PhysActivity', 0) == 0:
        risk_factors.append("Lack of physical activity")
    if blood_sample.get('HvyAlcoholConsump', 0) == 1:
        risk_factors.append("Heavy alcohol consumption")
    if blood_sample.get('GenHlth', 0) >= 4:
        risk_factors.append("Poor general health")
    if blood_sample.get('DiffWalk', 0) == 1:
        risk_factors.append("Difficulty walking")
    return risk_factors

def predict_diabetes(blood_sample: Dict[str, Any], model, preprocessor):
    risk_factors = get_risk_factors(blood_sample)
    ml_input = pd.DataFrame([blood_sample])
    ml_input = add_engineered_features(ml_input)
    ml_input_preprocessed = preprocessor.transform(ml_input)
    diabetes_probability = model.predict_proba(ml_input_preprocessed)[0][1]
    diabetes_probability = max(min(diabetes_probability, 0.99999), 0.00001)
    
    # Use the same threshold as in training (0.3)
    predicted_status = "At Risk" if diabetes_probability >= CLASSIFICATION_THRESHOLD else "Low Risk"
    
    if predicted_status == "At Risk":
        reason = f"Based on the analysis, there's a high likelihood of diabetes (probability: {diabetes_probability:.4f}). "
        if risk_factors:
            reason += f"Key contributing factors include: {', '.join(risk_factors)}. "
        suggestions = [
            "Consult with a healthcare provider for further evaluation and diagnosis.",
            "Consider lifestyle changes such as a balanced diet, regular exercise, and weight management.",
            "Monitor blood sugar levels regularly.",
            "Reduce stress and ensure adequate sleep.",
            "Explore diabetic management plans, if confirmed."
        ]
    else:
        reason = f"The analysis indicates a low risk of diabetes (probability: {diabetes_probability:.4f})."
        if risk_factors:
            reason += f" However, attention is advised for: {', '.join(risk_factors)}."
        suggestions = [
            "Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.",
            "Keep up with routine health check-ups to monitor key health indicators.",
            "Stay informed about diabetes prevention strategies.",
            "Maintain a healthy weight and avoid smoking.",
            "Ensure a diet rich in fruits, vegetables, and whole grains."
        ]
    
    return {
        "PredictedStatus": predicted_status,
        "DiabetesProbability": float(diabetes_probability),  # Convert to Python float
        "Reason": reason,
        "Suggestions": suggestions
    }

def generate_report(prediction: Dict[str, Any], blood_sample: Dict[str, Any]) -> str:
    report = f"Diabetes Risk Assessment Report\n\n"
    report += f"Predicted Status: {prediction['PredictedStatus']}\n"
    report += f"Diabetes Probability: {prediction['DiabetesProbability']:.2%}\n\n"
    report += f"Reason:\n{prediction['Reason']}\n\n"
    report += "Suggestions:\n"
    for suggestion in prediction['Suggestions']:
        report += f"- {suggestion}\n"
    report += "\nInput Features:\n"
    for key, value in blood_sample.items():
        report += f"{key}: {value}\n"
    return report