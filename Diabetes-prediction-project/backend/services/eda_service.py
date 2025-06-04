import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def perform_eda(df):
    
    # Convert relevant columns to numeric
    numeric_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    eda_results = {}

    # Ensure all columns are the correct data type
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Distribution of the target variable
    plt.figure(figsize=(10, 6))
    df['Diabetes_binary'].value_counts().plot(kind='bar')
    plt.title('Distribution of Diabetes Outcome')
    eda_results['target_distribution'] = plot_to_base64(plt)

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    eda_results['correlation_heatmap'] = plot_to_base64(plt)

    # Box plots for numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Diabetes_binary', y=feature, data=df)
        plt.title(f'Box Plot of {feature} by Diabetes Outcome')
        eda_results[f'boxplot_{feature}'] = plot_to_base64(plt)

    return eda_results

def plot_to_base64(plt):
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img