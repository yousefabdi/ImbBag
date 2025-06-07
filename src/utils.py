# utils.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np

def preprocess_data(X, y, test_size=0.2, random_state=None, stratify=True):
    """
    Preprocess data by splitting into train/test sets
    
    Parameters:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test samples (default: 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to preserve class distribution (default: True)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify = y if stratify else None
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=stratify,
        random_state=random_state
    )

def evaluate_model(model, X_test, y_test, verbose=True):
    """
    Evaluate model performance using multiple metrics
    
    Parameters:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
        verbose: Whether to print detailed report (default: True)
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    results = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    }
    
    if verbose:
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    
    return results