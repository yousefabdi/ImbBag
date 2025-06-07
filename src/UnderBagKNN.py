# UnderBagKNN.py
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class UnderBagKNN(ImbalanceBaggingClassifier):
    """
    Under-bagging KNN classifier for multi-class classification.
    
    This ensemble method creates bootstraps via a weighted resampling process.
    For each bootstrap, samples are drawn from a managed dataset built by
    pooling all classes with weights computed to reflect each class’s subsample rate.
    A KNeighborsClassifier is trained on each bootstrap sample.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, subsample_rate=1, n_neighbors=5, random_state=None):
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.subsample_rate = subsample_rate
        self.n_neighbors = n_neighbors
        self.base_estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        if random_state is not None:
            np.random.seed(random_state)

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "subsample_rate": self.subsample_rate,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            base_params = self.base_estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"base_estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in ["n_estimators", "subsample_rate", "n_neighbors", "random_state"]:
                setattr(self, parameter, value)
            elif parameter.startswith("base_estimator__"):
                param_name = parameter.split("__", 1)[1]
                self.base_estimator.set_params(**{param_name: value})
            else:
                raise ValueError(f"Invalid parameter {parameter} for UnderBagKNN")
        return self

    def __make_bootstraps(self, data):
        """
        Create bootstrap samples via weighted resampling.
        
        The method first computes the number of samples in the minority class (lowest_n)
        and uses it to determine a target number of samples per bootstrap:
        
            n_resamples = round(subsample_rate * lowest_n * n_classes)
        
        Then, for each class, weights are computed and all class data is aggregated into a
        managed dataset. Finally, weighted resampling (using np.random.choice) is performed to
        form each bootstrap sample.
        """
        classes, counts = np.unique(data[:, -1], return_counts=True)
        n_classes = len(classes)
        min_count = min(counts)
        
        # Adjust neighbors if minority class is too small
        if min_count < self.n_neighbors:
            warnings.warn(
                f"Minority class count ({min_count}) < n_neighbors ({self.n_neighbors}). "
                f"Reducing n_neighbors to {min_count}.",
                UserWarning
            )
            self.n_neighbors = min_count
            self.base_estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        n_resamples = int(round(self.subsample_rate * min_count * n_classes))
        weights = []
        managed_data = []
        
        for cls in classes:
            class_data = data[data[:, -1] == cls]
            class_weight = n_resamples / (n_classes * len(class_data))
            weights.extend([class_weight] * len(class_data))
            managed_data.append(class_data)
            
        managed_data = np.vstack(managed_data)
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize to probabilities
        
        boot_samples = {}
        for b in range(self.n_estimators):
            indices = np.random.choice(
                len(managed_data), 
                size=n_resamples, 
                replace=True, 
                p=weights
            )
            boot_samples[f"boot_{b}"] = {"boot": managed_data[indices]}
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
            model = clone(self.base_estimator)
            boot_data = dcBoot[b]["boot"]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self