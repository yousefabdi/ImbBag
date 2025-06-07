# ADASYNBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from imblearn.over_sampling import ADASYN
from sklearn.utils import check_random_state
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class ADASYNBag(ImbalanceBaggingClassifier):
    """
    ADASYNBagging classifier for binary classification.
    
    This estimator creates bootstrap samples with oversampling applied
    via ADASYN for each ensemble component.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, estimator=DecisionTreeClassifier(), 
                 n_neighbors=5, random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)
        self.n_neighbors = n_neighbors
        
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params.update({
            "n_neighbors": self.n_neighbors
        })
        if deep and hasattr(self.estimator, 'get_params'):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f'estimator__{key}'] = value
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
        minority_count = len(d_minority)
        idx_maj = np.where(data[:, -1] == Lmaj)[0]
        d_majority = data[idx_maj, :]

        if minority_count < 2:
            raise ValueError(
                "ADASYN requires at least 2 minority samples. "
                f"Only {minority_count} minority sample(s) available."
            )

        boot_dict = {}
        rng = check_random_state(self.random_state)
        
        for b in range(self.n_estimators):
            indices = np.arange(d_majority.shape[0])
            sampled_indices = rng.choice(indices, replace=True, size=len(indices))
            d_majority_sampled = d_majority[sampled_indices, :]
            D_temp = np.vstack([d_majority_sampled, d_minority])
            
            n_neighbors_eff = min(self.n_neighbors, minority_count - 1)
            
            try:
                ada = ADASYN(
                    random_state=rng.randint(1, 1000000) if self.random_state else None, 
                    n_neighbors=n_neighbors_eff
                )
                X_res, y_res = ada.fit_resample(D_temp[:, :-1], D_temp[:, -1])
                b_samp = np.column_stack((X_res, y_res))
            except ValueError as e:
                raise RuntimeError(
                    f"ADASYN failed with {minority_count} minority samples "
                    f"and {n_neighbors_eff} neighbors: {str(e)}"
                )
            
            boot_dict[f'boot_{b}'] = b_samp
        return boot_dict

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("ADASYNBagging is designed for binary classification only.")

        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        training_data = np.column_stack((X, y))
        boot_samples = self.__make_bootstraps(training_data)
        
        self.models = []
        for key in boot_samples:
            boot_data = boot_samples[key]
            model = clone(self.estimator)
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
        return self