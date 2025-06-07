# Metrics.py
import numpy as np

class HVDM:
    """
    Heterogeneous Value Difference Metric (HVDM) for mixed data types.
    
    Parameters:
        categorical_indices (list): Indices of categorical features
        numeric_indices (list): Indices of numeric features
    """
    def __init__(self, categorical_indices, numeric_indices):
        self.categorical_indices = categorical_indices
        self.numeric_indices = numeric_indices
        
    def fit(self, X, y):
        """
        Precompute statistics needed for distance calculation.
        
        Parameters:
            X (array-like): Training data
            y (array-like): Target values
        """
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.range_ = {}
        self.class_prior_ = np.array([np.mean(y == c) for c in self.classes_])
        self.categorical_probs_ = {}
        
        # Process numeric features
        for j in self.numeric_indices:
            col = X[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            r = max_val - min_val
            self.range_[j] = r if r != 0 else 1.0  # Avoid division by zero
        
        # Process categorical features
        for j in self.categorical_indices:
            col = X[:, j]
            unique_vals = np.unique(col)
            probs_dict = {}
            
            for val in unique_vals:
                mask = (col == val)
                y_sub = y[mask]
                probs = [np.mean(y_sub == c) for c in self.classes_]
                probs_dict[val] = np.array(probs)
                
            self.categorical_probs_[j] = probs_dict
        return self
    
    def distance(self, a, b):
        """
        Compute HVDM between two instances.
        
        Parameters:
            a (array-like): First instance
            b (array-like): Second instance
            
        Returns:
            float: HVDM distance
        """
        total_sq = 0.0
        
        # Numeric features
        for j in self.numeric_indices:
            if j < len(a) and j < len(b):  # Ensure index exists
                r = self.range_.get(j, 1.0)
                diff = np.abs(float(a[j]) - float(b[j]))
                total_sq += (diff / r) ** 2
        
        # Categorical features
        for j in self.categorical_indices:
            if j < len(a) and j < len(b):  # Ensure index exists
                val1 = a[j]
                val2 = b[j]
                
                if val1 == val2:
                    term = 0.0
                else:
                    probs_dict = self.categorical_probs_.get(j, {})
                    probs1 = probs_dict.get(val1, self.class_prior_)
                    probs2 = probs_dict.get(val2, self.class_prior_)
                    term = np.sum((probs1 - probs2) ** 2)
                total_sq += term
        
        return np.sqrt(total_sq)