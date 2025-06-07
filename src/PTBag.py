# PTBag.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

class PTBag(BaseEstimator, ClassifierMixin):
    """
    PT-Bagging classifier based on threshold-moving for imbalanced data.
    
    Source:
      Collell, G., Prelec, D., & Patil, K. R. (2018). A simple plug-in bagging ensemble based on
      threshold-moving for classifying binary and multiclass imbalanced data. Neurocomputing, 275, 330-340.
    
    This classifier first trains a Bagging ensemble. During prediction, the probability for
    each class is adjusted by dividing by the class prior (λₖ) and the class with the highest
    adjusted probability is returned.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.random_state = random_state
        self.ensemble = None
        self.lamda_k = None
        self.classes_ = None
        self.n_features_in_ = None

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "random_state": self.random_state
        }
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        for key, value in parameters.items():
            if key in ["n_estimators", "random_state"]:
                setattr(self, key, value)
            elif key == "estimator":
                setattr(self, key, value)
            elif key.startswith("estimator__"):
                param_name = key.split("__", 1)[1]
                self.estimator.set_params(**{param_name: value})
            else:
                setattr(self, key, value)
                
        # Reinitialize ensemble if parameters changed
        if "n_estimators" in parameters or "estimator" in parameters:
            self.ensemble = BaggingClassifier(
                estimator=self.estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Calculate class priors
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.lamda_k = {cls: count / total_samples for cls, count in zip(unique, counts)}
        
        # Initialize and fit ensemble
        if self.ensemble is None:
            self.ensemble = BaggingClassifier(
                estimator=self.estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels using threshold-moving.
        
        For each test sample, this method computes the averaged probability estimates from the ensemble,
        then adjusts each class probability by dividing by its prior (λₖ) and returns the class with
        the highest adjusted probability.
        """
        check_is_fitted(self, ['ensemble', 'lamda_k', 'classes_'])
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
            
        proba = self.predict_proba(X)
        adjusted_proba = np.zeros_like(proba)
        
        for i, cls in enumerate(self.classes_):
            adjusted_proba[:, i] = proba[:, i] / self.lamda_k[cls]
            
        return self.classes_[np.argmax(adjusted_proba, axis=1)]
    
    def predict_proba(self, X):
        check_is_fitted(self, ['ensemble'])
        X = check_array(X)
        return self.ensemble.predict_proba(X)