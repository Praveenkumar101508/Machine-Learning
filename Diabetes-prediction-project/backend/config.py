import os

# Base directory
BASE_DIR = '/Users/praveenkumar/Documents/Diabetes-prediction-project'

# Random state for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes_binary_health_indicators_BRFSS2015.csv')
PDF_DIR = os.path.join(BASE_DIR, 'dataset', 'Blood_Sample_Reports')
OUTPUT_GRAPHS = os.path.join(BASE_DIR, 'dataset', 'output_graphs')

# Ensure the OUTPUT_GRAPHS directory exists
os.makedirs(OUTPUT_GRAPHS, exist_ok=True)

# Model and variables paths
MODEL_PATH = os.path.join(OUTPUT_GRAPHS, 'diabetes_model.joblib')
VARIABLES_PATH = os.path.join(OUTPUT_GRAPHS, 'variables.joblib')

# Other constants
CLASSIFICATION_THRESHOLD = 0.3
ALLOWED_EXTENSIONS = {'pdf'}

# Graph file names
DISTRIBUTION_GRAPH = os.path.join(OUTPUT_GRAPHS, 'distribution_graph.png')
CORRELATION_HEATMAP = os.path.join(OUTPUT_GRAPHS, 'correlation_heatmap.png')
PAIRPLOT = os.path.join(OUTPUT_GRAPHS, 'pairplot.png')
FEATURE_IMPORTANCE = os.path.join(OUTPUT_GRAPHS, 'feature_importance.png')
MODEL_COMPARISON = os.path.join(OUTPUT_GRAPHS, 'model_comparison.png')

# Flask settings
DEBUG = True
PORT = 5000
