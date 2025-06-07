# SMOTEBag.py
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class SMOTEBag(ImbalanceBaggingClassifier):
    """
    SMOTE Bagging classifier for multi-class classification.
    
    This algorithm generates bootstrap samples using SMOTE oversampling.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, k_neighbors=5, estimator=DecisionTreeClassifier(), random_state=None):
        super().__init__(n_estimators=n_estimators, estimator=estimator, random_state=random_state)
        self.k_neighbors = k_neighbors

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "k_neighbors": self.k_neighbors,
            "estimator": self.estimator,
            "random_state": self.random_state
        }
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in ["n_estimators", "k_neighbors", "estimator", "random_state"]:
                setattr(self, parameter, value)
            elif parameter.startswith("estimator__"):
                param_name = parameter.split("__", 1)[1]
                self.estimator.set_params(**{param_name: value})
            else:
                raise ValueError(f"Invalid parameter {parameter} for estimator SMOTEBag")
        return self

    def __make_bootstraps(self, data):
        """
        Create bootstrap samples using SMOTE oversampling.
        
        For each bootstrap iteration, if the minimum class count (min_k_neighbors) is less than 6,
        then SMOTE is created with k_neighbors set to (min_k_neighbors - 1); otherwise, the provided
        k_neighbors value is used.
        """
        boot_samples = {}
        min_count = min(np.bincount(data[:, -1].astype(int)))
        
        for b in range(self.n_estimators):
            k_val = min(self.k_neighbors, min_count - 1) if min_count > 1 else 1
            smote = SMOTE(
                sampling_strategy="auto",
                k_neighbors=max(1, k_val),
                random_state=self.random_state
            )
            X_res, y_res = smote.fit_resample(data[:, :-1], data[:, -1])
            boot_samples[f"boot_{b}"] = {"boot": np.c_[X_res, y_res]}
        return boot_samples

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if len(np.unique(y)) < 2:
            raise ValueError("The data should have at least two classes.")
            
        self.classes_ = np.unique(y)
        training_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        dcBoot = self.__make_bootstraps(training_data)
        
        self.models = []
        for b in dcBoot:
            model = clone(self.estimator)
            boot_data = dcBoot[b]["boot"]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self