# BBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from collections import Counter

from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class BBag(ImbalanceBaggingClassifier):
    """
    Description: This file contains a class that implements "Boundary Bagging (BBag)" algorithm using scikit-learn.
    BBag was designed for multi-class datasets

    Source: Boukir, S., & Feng, W. (2021, January). Boundary bagging to address training data issues in ensemble classification. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 9975-9981). IEEE.

    Programmer: Yousef Abdi
    Email: yousef.abdi@gmail.com
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, estimator=DecisionTreeClassifier(), 
                 random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)

    def __make_bootstraps(self, data, counts, min_lbl, minority_count, margin):    
        boot_samples = {}    
        rng = check_random_state(self.random_state)
        
        for i in range(self.n_estimators):
            idx = np.where(data[:, -1] == min_lbl)[0]
            minority_class = data[idx, :]
            sampled_data = minority_class
            
            for label, count in counts.items():
                if label != min_lbl:
                    class_indices = np.where(data[:, -1] == label)[0]
                    class_data = data[class_indices, :]
                    class_margins = margin[class_indices]
                    rem_samples = len(class_indices) - minority_count
                    
                    if rem_samples > 0:
                        rand_number = rng.randint(
                            int(0.2 * rem_samples), 
                            rem_samples + 1
                        )
                    else:
                        rand_number = 0
                        
                    top_indices = np.argsort(class_margins)[-rand_number:] if rand_number > 0 else []
                    mask = np.ones(len(class_margins), dtype=bool)
                    if len(top_indices) > 0:
                        mask[top_indices] = False
                    new_class_data = class_data[mask]
                    sampled_data = np.concatenate((sampled_data, new_class_data), axis=0)
                    
            boot_samples[f'boot_{i}'] = sampled_data     
        return boot_samples
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        training_data = np.column_stack((X, y))
        
        counts = Counter(training_data[:, -1])
        minority_count = float('inf')
        for label, count in counts.items():
            if count < minority_count:
                minority_count = count
                min_lbl = label

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Create temporary ensemble for margin calculation
        init_ensemble = BaggingClassifier(
            estimator=clone(self.estimator),
            n_estimators=self.n_estimators,
            random_state=self.random_state
        ).fit(X, y)
        
        # Calculate margins
        margin = np.zeros(len(training_data))
        for idx, x in enumerate(training_data):
            votes = np.zeros(len(self.classes_))
            for estimator in init_ensemble.estimators_:
                pred = estimator.predict([x[:-1]])[0]
                class_idx = np.where(self.classes_ == pred)[0][0]
                votes[class_idx] += 1
            
            true_class_idx = np.where(self.classes_ == x[-1])[0][0]
            v_y = votes[true_class_idx]
            other_votes = votes.copy()
            other_votes[true_class_idx] = 0
            v_c = np.max(other_votes)
            
            margin[idx] = (v_y - v_c) / (v_y + v_c) if (v_y + v_c) != 0 else 0
        
        boot_samples = self.__make_bootstraps(
            training_data, counts, min_lbl, minority_count, margin
        )
    
        self.models = []
        for key in boot_samples:
            boot_data = boot_samples[key]
            model = clone(self.estimator)
            model.fit(boot_data[:, :-1], boot_data[:, -1])
            self.models.append(model)
    
        return self