import logging
import PyPDF2
from typing import Dict, Any
import re

# Get a logger instance
logger = logging.getLogger(__name__)

def extract_features_from_pdf(pdf_path: str) -> Dict[str, Any]:
    features = {}
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text
                logger.info(f"Extracted text from page: {page_text}")  # Use logger here
        
        logger.info(f"Full extracted text from PDF: {text}")  # Use logger here

        def safe_extract(pattern, text, default=0):
            match = re.search(pattern, text)
            return float(match.group(1)) if match else default

        features['BMI'] = safe_extract(r'BMI:\s*([\d.]+)', text)
        features['Age'] = safe_extract(r'Age:\s*([\d.]+)', text)
        features['HighBP'] = 1 if 'Blood Pressure: High' in text else 0
        features['HighChol'] = safe_extract(r'Cholesterol Levels:\s*([\d.]+)', text)
        features['Glucose'] = safe_extract(r'Glucose Levels:\s*([\d.]+)', text)
        features['HeartDiseaseorAttack'] = 1 if 'Heart Disease History: Yes' in text else 0
        features['PhysActivity'] = 0 if 'Physical Activity: No' in text else 1
        features['Fruits'] = safe_extract(r'Fruits:\s*([\d.]+)', text)
        features['Veggies'] = safe_extract(r'Veggies:\s*([\d.]+)', text)
        features['HvyAlcoholConsump'] = 1 if 'Heavy Alcohol Consumption: Yes' in text else 0
        features['AnyHealthcare'] = 1 if 'Healthcare Access: Yes' in text else 0
        features['NoDocbcCost'] = 1 if 'Cost Prevented Doctor Visit: Yes' in text else 0
        features['GenHlth'] = safe_extract(r'General Health:\s*([\d.]+)', text)
        features['MentHlth'] = safe_extract(r'Mental Health \(Days\):\s*([\d.]+)', text)
        features['PhysHlth'] = safe_extract(r'Physical Health \(Days\):\s*([\d.]+)', text)
        features['DiffWalk'] = 1 if 'Difficulty Walking: Yes' in text else 0
        features['Sex'] = 0 if 'Sex: Female' in text else 1
        features['Education'] = safe_extract(r'Education Level:\s*([\d.]+)', text)
        features['Income'] = safe_extract(r'Income Level:\s*([\d.]+)', text)
        features['Stroke'] = 1 if 'Stroke History: Yes' in text else 0
        features['Smoker'] = 1 if 'Smoking Status: Smoker' in text else 0
        features['CholCheck'] = 1  # Assuming cholesterol check was done for all patients

        logger.info(f"Extracted features: {features}")
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(f"Problematic text: {text}")
    
    return features
