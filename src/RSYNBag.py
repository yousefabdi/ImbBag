# RSYNBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class RSYNBag(ImbalanceBaggingClassifier):
    """
    RSYN Bagging classifier for binary classification.
    
    RSYNBag generates bootstrap samples via a resampling procedure that applies ADASYN oversampling
    on some iterations (when certain conditions are met), and otherwise simply combines resampled
    majority samples with the minority class.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, estimator=DecisionTreeClassifier(), n_neighbors=5, random_state=None):
        super().__init__(n_estimators=n_estimators, estimator=estimator, random_state=random_state)
        self.n_neighbors = n_neighbors
        if random_state is not None:
            np.random.seed(random_state)

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state
        }
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in ["n_estimators", "estimator", "n_neighbors", "random_state"]:
                setattr(self, parameter, value)
            elif parameter.startswith("estimator__"):
                param_name = parameter.split("__", 1)[1]
                self.estimator.set_params(**{param_name: value})
            else:
                raise ValueError(f"Invalid parameter {parameter} for estimator RSYNBag")
        return self

    def __make_bootstraps(self, data):
        classes, counts = np.unique(data[:, -1], return_counts=True)
        if len(classes) != 2:
            raise ValueError("RSYNBag only supports binary classification")
            
        minority_class = classes[0] if counts[0] < counts[1] else classes[1]
        majority_class = classes[1] if counts[0] < counts[1] else classes[0]
        
        minority_data = data[data[:, -1] == minority_class]
        majority_data = data[data[:, -1] == majority_class]
        
        boot_samples = {}
        for b in range(self.n_estimators):
            maj_sample = majority_data[np.random.choice(
                len(majority_data), 
                size=len(majority_data), 
                replace=True
            )]
            
            if b % 2 == 0 and (len(minority_data) / len(majority_data)) < 0.8 and len(minority_data) > 5:
                temp = np.vstack([maj_sample, minority_data])
                ada = ADASYN(
                    sampling_strategy='minority',
                    n_neighbors=min(self.n_neighbors, len(minority_data)-1),
                    random_state=self.random_state
                )
                X_res, y_res = ada.fit_resample(temp[:, :-1], temp[:, -1])
                boot_samples[f'boot_{b}'] = {'boot': np.c_[X_res, y_res]}
            else:
                boot_samples[f'boot_{b}'] = {'boot': np.vstack([maj_sample, minority_data])}
                
        return boot_samples

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        training_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        boot_samples = self.__make_bootstraps(training_data)
        
        self.models = []
        for b in boot_samples:
            model = clone(self.estimator)
            boot_data = boot_samples[b]['boot']
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self