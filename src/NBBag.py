# NBBag.py
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from Metrics import HVDM
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class NBBag(ImbalanceBaggingClassifier):
    """
    Neighborhood Balanced Bagging (NBBag) classifier for binary classification.
    
    Implements the algorithm from:
    Błaszczyński, J., & Stefanowski, J. (2015). Neighbourhood sampling in bagging for imbalanced data. 
    Neurocomputing, 150, 529-542.
    
    Parameters:
        n_estimators (int): Number of base estimators in the ensemble
        estimator (object): Base classifier (default: DecisionTreeClassifier)
        n_neighbors (int): Number of neighbors for neighborhood calculation
        phi (float): Exponent for neighborhood weighting (default: 0.5 for undersampling, 2 for oversampling)
        sampling_method (str): Resampling method - "undersampling" or "oversampling"
        categorical_indices (list): Indices of categorical features for HVDM
        numeric_indices (list): Indices of numeric features for HVDM
        random_state (int): Random seed for reproducibility
        
    Attributes:
        classes_ (array): Class labels
        estimators_ (list): Trained base classifiers
        n_features_in_ (int): Number of features seen during fit
        hvdm_ (HVDM): Prefitted HVDM metric calculator

    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, 
                 n_estimators=10, 
                 estimator=None,
                 n_neighbors=5, 
                 phi=None, 
                 sampling_method="undersampling",
                 categorical_indices=None, 
                 numeric_indices=None,
                 random_state=None):
        
        # Fix estimator initialization
        if estimator is None:
            estimator = DecisionTreeClassifier()
            
        super().__init__(n_estimators, estimator, random_state)
        self.n_neighbors = n_neighbors
        self.phi = phi
        self.sampling_method = sampling_method
        self.categorical_indices = categorical_indices or []
        self.numeric_indices = numeric_indices or []
        
        # Set default phi based on sampling method
        if self.phi is None:
            if self.sampling_method == "undersampling":
                self.phi = 0.5
            else:  # oversampling
                self.phi = 2
        
    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "n_neighbors": self.n_neighbors,
            "phi": self.phi,
            "sampling_method": self.sampling_method,
            "categorical_indices": self.categorical_indices,
            "numeric_indices": self.numeric_indices,
            "random_state": self.random_state
        }
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ["n_estimators", "n_neighbors", "phi", "sampling_method", 
                       "categorical_indices", "numeric_indices", "random_state"]:
                setattr(self, key, value)
            elif key == "estimator":
                self.estimator = value
            elif key.startswith("estimator__"):
                if hasattr(self.estimator, "set_params"):
                    self.estimator.set_params(**{key[11:]: value})
            else:
                raise ValueError(f"Invalid parameter {key} for estimator NBBag")
        return self
    
    def _precompute_weights(self, X, y):
        """Compute instance weights based on neighborhood class distribution"""
        # Create and fit HVDM metric
        self.hvdm_ = HVDM(
            categorical_indices=self.categorical_indices,
            numeric_indices=self.numeric_indices
        )
        self.hvdm_.fit(X, y)
        
        # Create custom metric function
        def hvdm_metric(a, b):
            return self.hvdm_.distance(a, b)
        
        # Find neighbors using HVDM
        knn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 to exclude self
            metric=hvdm_metric,
            n_jobs=-1
        )
        knn.fit(X)
        neighbors = knn.kneighbors(return_distance=False)
        
        # Remove self from neighbors
        neighbors = neighbors[:, 1:]
        
        # Get class info
        self.classes_ = np.unique(y)
        minority_class = self.classes_[0] if np.sum(y == self.classes_[0]) < np.sum(y == self.classes_[1]) else self.classes_[1]
        majority_class = self.classes_[1] if minority_class == self.classes_[0] else self.classes_[0]
        
        # Compute weights
        weights = np.zeros(len(y))
        minority_mask = (y == minority_class)
        majority_mask = (y == majority_class)
        
        # Majority instances get base weight
        weights[majority_mask] = 0.5 * (np.sum(minority_mask) / np.sum(majority_mask))
        
        # Minority instances get neighborhood-based weight
        for i in np.where(minority_mask)[0]:
            neighbor_classes = y[neighbors[i]]
            maj_count = np.sum(neighbor_classes == majority_class)
            L = (maj_count ** self.phi) / self.n_neighbors
            weights[i] = 0.5 * (L + 1)
            
        return weights, minority_class, majority_class
    
    def fit(self, X, y):
        """Fit the NBBag ensemble classifier"""
        # Validate inputs
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]  # Store number of features for scikit-learn
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NBBag requires binary classification data")
            
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Precompute weights
        weights, minority_class, majority_class = self._precompute_weights(X, y)
        
        # Get class counts
        minority_mask = (y == minority_class)
        majority_mask = (y == majority_class)
        n_minority = np.sum(minority_mask)
        n_majority = np.sum(majority_mask)
        
        # Create bootstrap samples
        self.models = []
        for _ in range(self.n_estimators):
            if self.sampling_method == "undersampling":
                # Undersampling: select n_minority samples from each class
                # Sample minority class
                min_indices = np.where(minority_mask)[0]
                min_selected = np.random.choice(
                    min_indices, 
                    size=n_minority, 
                    replace=True, 
                    p=weights[min_indices]/np.sum(weights[min_indices])
                )
                
                # Sample majority class
                maj_indices = np.where(majority_mask)[0]
                maj_selected = np.random.choice(
                    maj_indices, 
                    size=n_minority, 
                    replace=True, 
                    p=weights[maj_indices]/np.sum(weights[maj_indices])
                )
                
                # Combine samples
                sample_indices = np.concatenate([min_selected, maj_selected])
                
            else:  # oversampling
                # Oversampling: use all minority samples and resampled majority
                # Sample minority class with weights
                min_indices = np.where(minority_mask)[0]
                min_selected = np.random.choice(
                    min_indices, 
                    size=n_minority, 
                    replace=True, 
                    p=weights[min_indices]/np.sum(weights[min_indices])
                )
                
                # Sample majority class with weights
                maj_indices = np.where(majority_mask)[0]
                maj_selected = np.random.choice(
                    maj_indices, 
                    size=n_majority, 
                    replace=True, 
                    p=weights[maj_indices]/np.sum(weights[maj_indices])
                )
                
                # Combine samples
                sample_indices = np.concatenate([min_selected, maj_selected])
            
            # Create bootstrap sample
            X_boot = X[sample_indices]
            y_boot = y[sample_indices]
            
            # Train base estimator
            model = clone(self.estimator)
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        return self
