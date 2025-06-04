# backend/models/diabetes_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from config import *

class DiabetesPredictor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.best_model = None

    def load_data(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(DATA_PATH)
        print(f"Dataset shape: {self.data.shape}")

    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_not_accepted:
            self.data[column] = self.data[column].replace(0, np.NaN)
        
        print("Missing values after handling zeros:")
        print(self.data.isnull().sum())

    def feature_scaling(self):
        """Perform feature scaling using StandardScaler."""
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

    def feature_engineering(self):
        """Perform feature engineering if needed."""
        # Add any feature engineering steps here
        pass

    def train_test_split(self):
        """Split the data into training and testing sets."""
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

    def preprocess_data(self):
        """Run all preprocessing steps."""
        self.load_data()
        self.handle_missing_values()
        self.feature_scaling()
        self.feature_engineering()
        self.train_test_split()

        # Fit the preprocessor on the training data
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

    def train_models(self):
        """Train multiple models on the preprocessed data."""
        models = {
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'SVM': SVC(probability=True, random_state=RANDOM_STATE),
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

    def evaluate_models(self):
        """Evaluate all trained models and select the best one."""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred)
            }
            print(f"\n{name} Results:")
            for metric, value in results[name].items():
                print(f"{metric.capitalize()}: {value:.4f}")

        self.best_model = max(results, key=lambda x: results[x]['accuracy'])
        print(f"\nBest performing model: {self.best_model}")

        # Save model comparison graph
        self.plot_model_comparison(results)

    def plot_distribution(self):
        """Plot the distribution of the target variable."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Outcome', data=self.data)
        plt.title('Distribution of Diabetes Outcome')
        plt.savefig(DISTRIBUTION_GRAPH)
        plt.close()

    def plot_correlation_heatmap(self):
        """Plot the correlation heatmap."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(CORRELATION_HEATMAP)
        plt.close()

    def plot_pairplot(self):
        """Plot the pairplot."""
        sns.pairplot(self.data, hue='Outcome')
        plt.savefig(PAIRPLOT)
        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance of the best model."""
        if hasattr(self.models[self.best_model], 'feature_importances_'):
            importances = self.models[self.best_model].feature_importances_
            feature_names = self.data.drop('Outcome', axis=1).columns
            feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            feat_importances.plot(kind='bar')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(FEATURE_IMPORTANCE)
            plt.close()

    def plot_model_comparison(self, results):
        """Plot model comparison."""
        accuracies = [results[model]['accuracy'] for model in results]
        model_names = list(results.keys())
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies)
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(MODEL_COMPARISON)
        plt.close()

    def save_model(self):
        """Save the best model and preprocessor."""
        joblib.dump((self.models[self.best_model], self.preprocessor), MODEL_PATH)
        print(f"Best model ({self.best_model}) saved as '{MODEL_PATH}'")

    def run(self):
        """Run the entire pipeline."""
        self.preprocess_data()
        self.plot_distribution()
        self.plot_correlation_heatmap()
        self.plot_pairplot()
        self.train_models()
        self.evaluate_models()
        self.plot_feature_importance()
        self.save_model()

# Usage
if __name__ == "__main__":
    predictor = DiabetesPredictor()
    predictor.run()