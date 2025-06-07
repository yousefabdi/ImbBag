# BEV.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from sklearn.utils import check_random_state
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class BEV(ImbalanceBaggingClassifier):
    """
    Bagging Ensemble Variation (BEV) classifier for binary datasets.

    Source:
        Li, C. (2007, March). Classifying imbalanced data using a bagging ensemble variation (BEV).
        In Proceedings of the 45th annual southeast regional conference (pp. 203-208).

    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, estimator=DecisionTreeClassifier(random_state=42), 
                 random_state=None):
        super().__init__(n_estimators=1,  # Will be updated during fit
                         estimator=estimator, 
                         random_state=random_state)
        self.baseclassifier = estimator

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params["estimator"] = self.baseclassifier
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
            Lmaj, Lmin = classes[0], classes[1]
        else:
            Lmaj, Lmin = classes[1], classes[0]

        idx_min = np.where(data[:, -1] == Lmin)[0]
        d_minority = data[idx_min, :]
        idx_maj = np.where(data[:, -1] == Lmaj)[0]
        d_majority = data[idx_maj, :]

        # Compute number of bootstrap sets
        n_sets = int(np.ceil(len(d_majority) / len(d_minority)))
        majority_sets = np.array_split(d_majority, n_sets)

        dc = {}
        for i, maj_set in enumerate(majority_sets):
            b_samp = np.vstack([maj_set, d_minority])
            dc[f'boot_{i}'] = b_samp
        return dc, n_sets

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("BEV classifier requires exactly two classes")
            
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        
        training_data = np.column_stack((X, y))
        boot_samples, n_sets = self.__make_bootstraps(training_data)
        
        # Update n_estimators based on data
        self.n_estimators = n_sets
        
        self.models = []
        for key in boot_samples:
            boot_data = boot_samples[key]
            model = clone(self.estimator)
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
            
        return self