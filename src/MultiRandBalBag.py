# MultiRandBalBag.py
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class MultiRandBalBag(ImbalanceBaggingClassifier):
    """
    Random Balance ensembles for multiclass imbalance learning (MultiRandBalBag)
    classifier for multi-class classification.
    
    This method uses a randomized balancing procedure for the training samples.
    
    Source:
      Rodríguez, J. J., Diez-Pastor, J. F., Arnaiz-Gonzalez, A., & Kuncheva, L. I. (2020).
      Random balance ensembles for multiclass imbalance learning. Knowledge-Based Systems, 193, 105434.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 k_neighbors=5, random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)
        self.K = k_neighbors

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params["k_neighbors"] = self.K
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        estimator_params = {}
        for key, value in parameters.items():
            if key == "k_neighbors":
                setattr(self, key, value)
            elif key.startswith("estimator__"):
                estimator_params[key.split('__', 1)[1]] = value
            else:
                setattr(self, key, value)
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        return self

    def __custom_smote(self, data, count, k_neighbors, rng):
        if len(data) <= 1:
            # Can't generate neighbors with < 2 samples
            return data[rng.choice(len(data), size=count, replace=True)]
        
        k = min(k_neighbors, len(data) - 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        new_samples = []
        while len(new_samples) < count:
            idx = rng.randint(len(data))
            neighbor_idx = rng.choice(indices[idx][1:])  # Skip self
            diff = data[neighbor_idx] - data[idx]
            gap = rng.uniform(0, 1)
            new_sample = data[idx] + gap * diff
            new_samples.append(new_sample)
            
        return np.array(new_samples)

    def __random_balance_bootstraps(self, training_data, rng):
        """
        Create a balanced bootstrap sample from training_data.
        The method computes balancing weights randomly across classes, and for each class:
          - If the desired count (ni) is less than or equal to the available samples, sample without replacement.
          - Otherwise, apply custom SMOTE to generate additional samples.
        """
        classes = np.unique(training_data[:, -1])
        n_classes = len(classes)
        
        # Generate random weights for each class
        w = rng.rand(n_classes)
        s_w = np.sum(w)
        X_balanced = []
        
        for i, cls in enumerate(classes):
            idx = np.where(training_data[:, -1] == cls)[0]
            class_data = training_data[idx]
            
            # Calculate desired sample size for this class (at least 2)
            ni = max(math.floor(len(training_data) * w[i] / s_w), 2)
            
            if ni <= len(class_data):
                # Sample without replacement
                sample_idx = rng.choice(len(class_data), size=ni, replace=False)
                X_balanced.append(class_data[sample_idx])
            else:
                # First use all available samples
                X_balanced.append(class_data)
                
                # Generate synthetic samples for the difference
                synthetic_count = ni - len(class_data)
                synthetic = self.__custom_smote(
                    class_data[:, :-1], 
                    synthetic_count, 
                    min(self.K, len(class_data) - 1),
                    rng
                )
                # Add labels to synthetic samples
                synthetic = np.column_stack((synthetic, np.full(synthetic_count, cls)))
                X_balanced.append(synthetic)
                
        return np.vstack(X_balanced)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Adjust K if needed
        counts = np.bincount(y)
        if min(counts) < self.K:
            self.K = max(1, min(counts))
            
        training_data = np.column_stack((X, y))
        rng = check_random_state(self.random_state)
        
        self.models = []
        for i in range(self.n_estimators):
            boot_sample = self.__random_balance_bootstraps(training_data, rng)
            model = clone(self.estimator)
            model.fit(boot_sample[:, :-1], boot_sample[:, -1])
            self.models.append(model)
            
        return self