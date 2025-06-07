# LazyBag.py
import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats

class LazyBag(BaseEstimator, ClassifierMixin):
    """
    Lazy Bagging (LazyBag) classifier for multi-class classification.
    
    Based on:
      Zhu, X. (2007, October). Lazy bagging for classifying imbalanced data.
      In Seventh IEEE International Conference on Data Mining (ICDM 2007) (pp. 763-768). IEEE.
    
    Unlike typical bagging ensemble methods, LazyBag does not pre-fit a global ensemble.
    Instead, it stores the training data (and a corresponding weight vector) during fit,
    and for each test instance creates multiple bootstrap samples on the fly (by combining 
    random samples from the training data with random samples from the test instance’s 
    K-nearest neighbors) to generate predictions via majority voting.
    
    Note: This algorithm does not support predict_proba.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 n_neighbors=5, beta=0.99, random_state=None):
        self.n_estimators = n_estimators
        self.beta = beta
        self.estimator = estimator
        self.K = n_neighbors
        self.random_state = random_state
        self.X_train_ = None
        self.y_train_ = None
        self.weights_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "n_neighbors": self.K,
            "beta": self.beta,
            "estimator": self.estimator,
            "random_state": self.random_state
        }
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        estimator_params = {}
        for key, value in parameters.items():
            if key.startswith("estimator__"):
                estimator_params[key.split('__', 1)[1]] = value
            else:
                setattr(self, key, value)
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        return self

    def __calculate_ir(self, data):
        """
        Calculate a weight (information-related) for each feature based on entropy.
        (This is a heuristic approximation of an Information-gain Ratio.)
        """
        ir_values = []
        for col in range(data.shape[1] - 1):  # Exclude target column
            unique_vals, counts = np.unique(data[:, col], return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            ir_values.append(entropy)
            
        # Normalize to [0,1]
        min_ir = min(ir_values)
        max_ir = max(ir_values)
        if max_ir - min_ir < 1e-10:
            return np.ones(len(ir_values))
        return [(ir - min_ir) / (max_ir - min_ir) for ir in ir_values]

    def fit(self, X, y):
        """
        Fit LazyBag by storing training data and computing feature weights.
        
        Parameters:
          X: array-like of shape (n_samples, n_features)
          y: array-like of shape (n_samples,)
        """
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        training_data = np.column_stack((X, y))
        self.weights_ = self.__calculate_ir(training_data)
        
        # Adjust K based on training size and beta
        if self.beta >= 1:
            self.K = len(X)
        else:
            omega = math.log(len(X)**(1 - self.beta), 4)
            self.K = max(1, min(len(X), round(len(X) * omega)))
            
        return self

    def __make_bootstrapsamp_fit(self, test_instance, data, weights, rng):
        """
        For a given test instance, generate an ensemble of predictions by creating
        multiple bootstrap samples. Each sample is formed by:
          - Drawing a bootstrap sample B from the training data (excluding the K nearest neighbors),
          - Drawing a bootstrap sample P from the K nearest neighbors of the test instance,
          - Combining B and P,
          - Training a clone of the base classifier on the combined sample to predict the test instance.
        """
        classes, counts = np.unique(data[:, -1], return_counts=True)
        K_local = self.K if min(counts) >= self.K else int(min(counts))
        
        knn = KNeighborsClassifier(
            n_neighbors=K_local,
            weights='distance',
            metric='wminkowski',
            metric_params={'w': weights},
            n_jobs=-1
        )
        knn.fit(data[:, :-1], data[:, -1])
        neighbors = knn.kneighbors([test_instance], return_distance=False)
        S = data[neighbors[0]]
        
        predictions = []
        for _ in range(self.n_estimators):
            # Bootstrap sample from entire dataset
            idx_all = np.arange(data.shape[0])
            sidx_B = rng.choice(idx_all, size=len(data) - K_local, replace=True)
            B = data[sidx_B]
            
            # Bootstrap sample from neighbors
            idx_S = np.arange(S.shape[0])
            sidx_P = rng.choice(idx_S, size=K_local, replace=True)
            P = S[sidx_P]
            
            b_samp = np.vstack([B, P])
            model = clone(self.estimator)
            model.fit(b_samp[:, :-1], b_samp[:, -1])
            predictions.append(model.predict([test_instance])[0])
            
        return predictions

    def predict(self, X):
        """
        Predict class labels for X by applying Lazy Bagging.
        
        For each test instance, form an ensemble of predictions using on‑demand
        bootstrapping and then return the majority vote.
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'weights_'])
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
            
        training_data = np.column_stack((self.X_train_, self.y_train_))
        rng = check_random_state(self.random_state)
        
        predictions = np.zeros((len(X), self.n_estimators))
        for i, test_instance in enumerate(X):
            preds = self.__make_bootstrapsamp_fit(test_instance, training_data, 
                                                self.weights_, rng)
            predictions[i] = preds
            
        ypred, _ = stats.mode(predictions, axis=1)
        return ypred.ravel()

    def predict_proba(self, X):
        raise NotImplementedError("LazyBag does not support probability prediction")