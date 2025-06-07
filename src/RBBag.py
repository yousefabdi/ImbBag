# RBBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class RBBag(ImbalanceBaggingClassifier):
    """
    Roughly Balanced Bagging (RBBag) classifier for binary classification.
    
    RBBag is designed for binary problems. It generates bootstrap samples by 
    (a) sampling the majority class using a negative binomial distribution and 
    (b) sampling the minority class using random choice.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=50, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
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

    def __make_bootstraps(self, data, rng):
        """
        Create bootstrap samples as follows:
          - Identify the majority (Lmaj) and minority (Lmin) classes.
          - For each bootstrap, sample from the majority class using a negative binomial
            distribution (the sample size is randomly drawn) and sample all minority data
            via random choice.
        
        Returns a dictionary of bootstrap samples.
        """
        classes, counts = np.unique(data[:, -1], return_counts=True)
        if len(classes) != 2:
            raise ValueError("RBBag requires exactly two classes")
            
        if counts[0] > counts[1]:
            Lmaj, Lmin = classes[0], classes[1]
            Nmaj, Nmin = counts[0], counts[1]
        else:
            Lmaj, Lmin = classes[1], classes[0]
            Nmaj, Nmin = counts[1], counts[0]
            
        idx_min = np.where(data[:, -1] == Lmin)[0]
        d_minority = data[idx_min]
        idx_maj = np.where(data[:, -1] == Lmaj)[0]
        d_majority = data[idx_maj]
        
        dc = {}
        for b in range(self.n_estimators):
            # Sample majority class with negative binomial distribution
            size = rng.negative_binomial(Nmin, 0.5)
            sidx_maj = rng.choice(len(d_majority), size=min(size, len(d_majority)), replace=True)
            
            # Sample minority class with replacement
            sidx_min = rng.choice(len(d_minority), size=len(d_minority), replace=True)
            
            b_samp = np.vstack([
                d_majority[sidx_maj],
                d_minority[sidx_min]
            ])
            dc[f"boot_{b}"] = b_samp
            
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("RBBag requires exactly two classes")
            
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        training_data = np.column_stack((X, y))
        rng = check_random_state(self.random_state)
        
        boot_samples = self.__make_bootstraps(training_data, rng)
        self.models = []
        
        for key in boot_samples:
            model = clone(self.estimator)
            boot_data = boot_samples[key]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
            
        return self