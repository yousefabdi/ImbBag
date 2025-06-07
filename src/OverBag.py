# OverBag.py
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class OverBag(ImbalanceBaggingClassifier):
    """
    OverBagging (OverBag) classifier for binary classification.
    
    This ensemble method uses oversampling (via RandomOverSampler)
    to generate balanced bootstrap samples and then builds an ensemble.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)

    def __make_bootstraps(self, data, rng):
        """
        Create bootstrap samples using oversampling.
        
        For each bootstrap iteration, RandomOverSampler is used to
        oversample the minority class, producing balanced samples.
        """
        dc = {}
        for b in range(self.n_estimators):
            seed = rng.randint(0, 1000000)
            ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
            X_res, y_res = ros.fit_resample(data[:, :-1], data[:, -1])
            b_samp = np.column_stack((X_res, y_res))
            dc[f"boot_{b}"] = b_samp
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("Dataset must have at least two classes")
            
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