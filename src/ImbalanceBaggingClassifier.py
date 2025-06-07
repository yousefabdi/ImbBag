import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats

class ImbalanceBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.random_state = random_state
        self.models = []  

    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement the fit method!")
    
    def predict(self, X):
        check_is_fitted(self, 'models')
        X = check_array(X)
        predictions = np.asarray([model.predict(X) for model in self.models])
        mode_prediction, _ = stats.mode(predictions, axis=0, keepdims=True)
        return mode_prediction.ravel()
    
    def predict_proba(self, X):
        check_is_fitted(self, 'models')
        X = check_array(X)
        probas = np.asarray([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)