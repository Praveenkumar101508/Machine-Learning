import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import RANDOM_STATE, DATA_PATH, PDF_DIR, OUTPUT_GRAPHS
import re
from sklearn.pipeline import Pipeline
import logging
import math
import os
import PyPDF2
from flask import current_app
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from services.utils import extract_features_from_pdf
from utils.helpers import print_colored, create_output_dir, save_plot

def extract_features_from_pdf(pdf_path: str) -> Dict[str, Any]:
    features = {}
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text
                current_app.logger.info(f"Extracted text from page: {page_text}")
        
        current_app.logger.info(f"Full extracted text from PDF: {text}")

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
        features['CholCheck'] = 1  

        current_app.logger.info(f"Extracted features: {features}")
    except Exception as e:
        current_app.logger.error(f"Error extracting features: {str(e)}")
        current_app.logger.error(f"Problematic text: {text}")
    
    return features

def plot_feature_histograms(data):
    create_output_dir()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 4
    n_rows = math.ceil(len(numeric_columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.histplot(data=data, x=column, hue='Diabetes_binary', kde=True, ax=axes[i], palette="Set2")
        axes[i].set_title(f'Distribution of {column}', fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        if len(data['Diabetes_binary'].unique()) > 1:
            axes[i].legend(title='Diabetes Status', loc='upper right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_plot(fig, 'feature_histograms.png')
    plt.show()

def plot_feature_boxplots(data):
    create_output_dir()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 4
    n_rows = math.ceil(len(numeric_columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.boxplot(data=data, x='Diabetes_binary', y=column, ax=axes[i], palette="Set3")
        axes[i].set_title(f'Boxplot of {column} by Diabetes Status', fontsize=14)
        axes[i].set_xlabel('Diabetes Status', fontsize=12)
        axes[i].set_ylabel(column, fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_plot(fig, 'feature_boxplots.png')
    plt.show()

def plot_correlation_heatmap(data):
    create_output_dir()
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    save_plot(fig, 'correlation_heatmap.png')
    plt.show()

def plot_pairplot(data):
    create_output_dir()
    selected_features = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth']
    sns.pairplot(data, hue='Diabetes_binary', vars=selected_features, palette="husl")
    plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
    save_plot(plt.gcf(), 'pairplot.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    plt.tight_layout()
    save_plot(fig, f'confusion_matrix_{model_name}.png')
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f'ROC Curve for {model_name}', fontsize=16)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_plot(fig, f'roc_curve_{model_name}.png')
    plt.show()

def load_and_preprocess_data(file_path=DATA_PATH, test_size=0.3, skip_eda=True):
    data = pd.read_csv(file_path)
    data = data.sample(frac=0.7, random_state=RANDOM_STATE)
    print_colored(f"Dataset shape: {data.shape}", "yellow")

    data = add_engineered_features(data)

    if not skip_eda:
        plot_feature_histograms(data)
        plot_feature_boxplots(data)
        plot_correlation_heatmap(data)
        plot_pairplot(data)

    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ]), categorical_features)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)

    print_colored(f"Training set shape after SMOTE and preprocessing: {X_train_balanced.shape}", "yellow")
    print_colored(f"Testing set shape after preprocessing: {X_test_preprocessed.shape}", "yellow")

    return X_train_balanced, X_test_preprocessed, y_train_balanced, y_test, preprocessor, data

def add_engineered_features(data):
    if isinstance(data, pd.DataFrame):
        if 'BMI' in data.columns and 'Age' in data.columns:
            data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
            data['Age_Category'] = pd.cut(data['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
            data['BMI_Age_Interaction'] = data['BMI'] * data['Age']
        
        if 'GenHlth' in data.columns and 'MentHlth' in data.columns and 'PhysHlth' in data.columns:
            data['Health_Score'] = data['GenHlth'] + (30 - data['MentHlth']) + (30 - data['PhysHlth'])
        
        if all(col in data.columns for col in ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Smoker']):
            data['Lifestyle_Score'] = data['PhysActivity'] + data['Fruits'] + data['Veggies'] - data['HvyAlcoholConsump'] - data['Smoker']
    elif isinstance(data, pd.Series):
        if 'BMI' in data.index and 'Age' in data.index:
            data['BMI_Category'] = pd.cut([data['BMI']], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])[0]
            data['Age_Category'] = pd.cut([data['Age']], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])[0]
            data['BMI_Age_Interaction'] = data['BMI'] * data['Age']
        
        if 'GenHlth' in data.index and 'MentHlth' in data.index and 'PhysHlth' in data.index:
            data['Health_Score'] = data['GenHlth'] + (30 - data['MentHlth']) + (30 - data['PhysHlth'])
        
        if all(col in data.index for col in ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Smoker']):
            data['Lifestyle_Score'] = data['PhysActivity'] + data['Fruits'] + data['Veggies'] - data['HvyAlcoholConsump'] - data['Smoker']
    
    return data

