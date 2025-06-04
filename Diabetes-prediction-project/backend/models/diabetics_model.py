from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from config import RANDOM_STATE

class CustomHistGradientBoostingClassifier(HistGradientBoostingClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y):
        super().fit(X, y)
        self.feature_importances_ = permutation_importance(self, X, y, n_repeats=10, random_state=RANDOM_STATE).importances_mean

def create_models():
    """
    Create a dictionary of machine learning models for diabetes prediction.
    
    Returns:
        dict: A dictionary containing initialized model objects.
    """
    models = {
        'hist_gradient_boosting': CustomHistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=20,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE
        ),
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_STATE
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'sgd_svm': SGDClassifier(
        loss='log_loss',  # Changed from 'hinge' to 'log_loss'
        penalty='l2',
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE
    )
    }
    
    return models
