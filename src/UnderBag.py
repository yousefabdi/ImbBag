# UnderBag.py
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class UnderBag(ImbalanceBaggingClassifier):
    """
    Random Under Bagging (UnderBag) classifier for multi-class classification.
    
    This ensemble method uses undersampling of the majority class (via RandomUnderSampler)
    to create balanced bootstrap samples and trains an ensemble of base classifiers.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, estimator=DecisionTreeClassifier(), random_state=None):
        super().__init__(n_estimators=n_estimators, estimator=estimator, random_state=random_state)

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
        for parameter, value in parameters.items():
            if parameter in ["n_estimators", "estimator", "random_state"]:
                setattr(self, parameter, value)
            elif parameter.startswith("estimator__"):
                param_name = parameter.split("__", 1)[1]
                self.estimator.set_params(**{param_name: value})
            else:
                raise ValueError(f"Invalid parameter {parameter} for UnderBag")
        return self

    def __make_bootstraps(self, data):
        """
        Generate bootstrap samples via RandomUnderSampler.
        
        For each bootstrap iteration, the majority class is undersampled
        (using sampling_strategy='majority') to balance the dataset.
        """
        boot_samples = {}
        classes, counts = np.unique(data[:, -1], return_counts=True)
        min_count = min(counts)
        
        for b in range(self.n_estimators):
            rus = RandomUnderSampler(
                sampling_strategy={c: min_count for c in classes},
                random_state=self.random_state + b if self.random_state is not None else None
            )
            X_res, y_res = rus.fit_resample(data[:, :-1], data[:, -1])
            boot_samples[f"boot_{b}"] = {"boot": np.c_[X_res, y_res]}
        return boot_samples

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("The dataset must have at least two classes.")
            
        training_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        dcBoot = self.__make_bootstraps(training_data)
        
        self.models = []
        for b in dcBoot:
            model = clone(self.estimator)
            boot_data = dcBoot[b]["boot"]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self