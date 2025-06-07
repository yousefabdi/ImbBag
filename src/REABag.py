# REABag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y
from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class REABag(ImbalanceBaggingClassifier):
    """
    Resampling Ensemble (REABag) classifier for multi-class imbalance learning.
    
    Source:
      Qian, Y., Liang, Y., Li, M., Feng, G., & Shi, X. (2014). A resampling ensemble
      algorithm for classification of imbalance problems. Neurocomputing, 143, 57-67.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, estimator=DecisionTreeClassifier(), k=2, random_state=None):
        super().__init__(n_estimators=n_estimators, estimator=estimator, random_state=random_state)
        self.k = k
        if random_state is not None:
            np.random.seed(random_state)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "k": self.k,
            "random_state": self.random_state
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in ["n_estimators", "estimator", "k", "random_state"]:
                setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter {parameter} for estimator REABag")
        return self

    def __make_bootstraps(self, data):
        """
        Create bootstrap samples as per the REABag algorithm.
        
        For each class, compute a resampling factor S based on its count and
        the ratios between the minority and majority class counts. Then, for each bootstrap,
        oversample (or undersample) the class accordingly.
        """
        classes, counts = np.unique(data[:, -1], return_counts=True)
        n_lowest = min(counts)
        n_largest = max(counts)
        R = n_lowest / n_largest
        alpha = -0.097 * R + 1.428
        beta = 0.198 * R + 0.738
        S = np.empty(len(classes))
        for i in range(len(classes)):
            t1 = (beta * n_largest - alpha * n_lowest) / (n_largest - n_lowest)
            t2 = ((alpha - beta) * n_largest * n_lowest) / ((n_largest - n_lowest) * counts[i])
            S[i] = t1 + t2
        
        boot_samples = {}
        for b in range(self.n_estimators):
            resampled_data = []
            for i, cls in enumerate(classes):
                idx = np.where(data[:, -1] == cls)[0]
                class_data = data[idx]
                target_size = int(round(counts[i] * S[i]))
                
                if target_size > len(class_data):  # Oversample
                    n_new = target_size - len(class_data)
                    new_samples = []
                    for _ in range(n_new):
                        center_idx = np.random.randint(0, len(class_data))
                        center = class_data[center_idx]
                        if len(class_data) > self.k:
                            knn = NearestNeighbors(n_neighbors=self.k)
                            knn.fit(class_data[:, :-1])
                            _, indices = knn.kneighbors([center[:-1]])
                            neighbors = class_data[indices[0]]
                        else:
                            neighbors = class_data
                        weights = np.random.rand(len(neighbors))
                        weights /= weights.sum()
                        new_sample = np.average(neighbors, axis=0, weights=weights)
                        new_samples.append(new_sample)
                    resampled_data.append(np.vstack([class_data, new_samples]))
                else:  # Undersample
                    selected_indices = np.random.choice(
                        len(class_data), 
                        size=target_size, 
                        replace=False
                    )
                    resampled_data.append(class_data[selected_indices])
            
            boot_samples[f"boot_{b}"] = {"boot": np.vstack(resampled_data)}
        return boot_samples

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if len(np.unique(y)) < 2:
            raise ValueError("The dataset must have at least two classes.")
            
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