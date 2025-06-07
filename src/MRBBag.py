# MRBBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class MRBBag(ImbalanceBaggingClassifier):
    """
    Multi-class Roughly Balanced Bagging (MRBBag) classifier.
    
    MRBBag is designed for multi-class classification and supports two sampling modes:
      - undersampling (default)
      - oversampling
      
    In undersampling mode, bootstrap samples are created so that each class is sampled
    roughly at the level of the minority class. In oversampling mode, each class is sampled
    up to the level of the majority class.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 sampling_method="undersampling", random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)
        self.sampling_method = sampling_method.lower()

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params["sampling_method"] = self.sampling_method
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        estimator_params = {}
        for key, value in parameters.items():
            if key == "sampling_method":
                setattr(self, key, value.lower())
            elif key.startswith("estimator__"):
                estimator_params[key.split('__', 1)[1]] = value
            else:
                setattr(self, key, value)
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        return self

    def __make_bootstraps(self, data):
        """
        Create bootstrap samples using the Roughly Balanced Bagging approach.
        
        For multi-class datasets, this method computes the minimum class count 
        ("lowest") and the maximum class count ("largest"), and it computes class
        sampling probabilities that are uniform. In undersampling mode, it draws a 
        sample of size equal to the smallest class count from each class (adjusting if needed);
        in oversampling mode, it draws a sample up to the largest class size.
        
        Returns a dictionary of bootstrap samples.
        """
        classes, counts = np.unique(data[:, -1], return_counts=True)
        n_classes = len(classes)
        rng = check_random_state(self.random_state)
        dc = {}
        
        if self.sampling_method == "undersampling":
            target_size = min(counts)
            probabilities = np.ones(n_classes) / n_classes
            
            for b in range(self.n_estimators):
                # Find a sample size where no class gets 0 samples
                current_target = target_size
                while True:
                    multinomial_samples = rng.multinomial(current_target, probabilities)
                    if np.all(multinomial_samples > 0):
                        break
                    current_target += 1
                
                b_samp = []
                for j, cls in enumerate(classes):
                    idx = np.where(data[:, -1] == cls)[0]
                    class_data = data[idx]
                    sample_size = min(multinomial_samples[j], len(class_data))
                    sidx = rng.choice(len(class_data), size=sample_size, replace=True)
                    b_samp.append(class_data[sidx])
                
                dc[f"boot_{b}"] = np.vstack(b_samp)
                
        else:  # Oversampling mode
            target_size = max(counts)
            probabilities = np.ones(n_classes) / n_classes
            
            for b in range(self.n_estimators):
                multinomial_samples = rng.multinomial(target_size, probabilities)
                b_samp = []
                for j, cls in enumerate(classes):
                    idx = np.where(data[:, -1] == cls)[0]
                    class_data = data[idx]
                    sample_size = min(multinomial_samples[j], len(class_data))
                    sidx = rng.choice(len(class_data), size=sample_size, replace=True)
                    b_samp.append(class_data[sidx])
                
                dc[f"boot_{b}"] = np.vstack(b_samp)
                
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("The dataset must have at least two classes.")
            
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        training_data = np.column_stack((X, y))
        boot_samples = self.__make_bootstraps(training_data)
        
        self.models = []
        for key in boot_samples:
            model = clone(self.estimator)
            boot_data = boot_samples[key]
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
            
        return self