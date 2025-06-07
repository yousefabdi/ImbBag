import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class EBBag(ImbalanceBaggingClassifier):
    """
    Exactly Balanced Bagging (EBBag) classifier for binary classification.
    
    EBBag is designed to work on binary problems by splitting the majority class 
    into Q subsets (where Q = ceil(max(class counts)/min(class counts))) and pairing each 
    with all minority samples to form balanced bootstrap samples.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42),
                 random_state=None):
        super().__init__(n_estimators=n_estimators, estimator=estimator, random_state=random_state)

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params['estimator'] = self.estimator
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

    def __make_bootstraps(self, data):
        classes, counts = np.unique(data[:, -1], return_counts=True)
        if counts[0] > counts[1]:
            Lmaj = classes[0]
            Lmin = classes[1]
        else:
            Lmaj = classes[1]
            Lmin = classes[0]
        Q = int(np.ceil(max(counts) / min(counts)))
        idx_maj = np.where(data[:, -1] == Lmaj)[0]
        majority_data = data[idx_maj, :]
        idx_min = np.where(data[:, -1] == Lmin)[0]
        minority_data = data[idx_min, :]
        majority_subsets = np.array_split(majority_data, Q)
        dc = {}
        for i, subset in enumerate(majority_subsets):
            bootstrap = np.concatenate((subset, minority_data), axis=0)
            dc[f'boot_{i}'] = bootstrap
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if unique_classes.size != 2:
            raise ValueError("EBBag requires exactly two classes")
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        training_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        boot_samples = self.__make_bootstraps(training_data)
        self.models = []
        for key in boot_samples:
            model = clone(self.estimator)
            boot_data = boot_samples[key]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self