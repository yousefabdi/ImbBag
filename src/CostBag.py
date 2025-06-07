import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.utils.validation import check_X_y, check_random_state
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class CostBag(ImbalanceBaggingClassifier):
    """
    Cost-sensitive Bagging (CostBag) classifier for multi-class classification.
    
    This method uses cost-sensitive resampling (via stratified bootstrap samples)
    to address class imbalance.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10,
                 estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'),
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
        rng = check_random_state(self.random_state)
        dc = {}
        for b in range(self.n_estimators):
            seed = rng.randint(np.iinfo(np.int32).max)
            X_res, y_res = resample(data[:, :-1], data[:, -1],
                                    replace=True,
                                    stratify=data[:, -1],
                                    random_state=seed)
            b_samp = np.concatenate((X_res, y_res.reshape(-1, 1)), axis=1)
            dc[f"boot_{b}"] = b_samp
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        training_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        boot_samples = self.__make_bootstraps(training_data)
        self.models = []
        for b in boot_samples:
            model = clone(self.estimator)
            boot_data = boot_samples[b]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self

