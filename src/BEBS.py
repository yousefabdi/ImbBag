# BEBS.py
import numpy as np
from sklearn.base import clone
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from sklearn.neighbors import NearestNeighbors
from imblearn.metrics import geometric_mean_score
from collections import Counter

from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class BEBS(ImbalanceBaggingClassifier):
    """
    Bagging of Extrapolation‐SMOTE SVM (BEBS) classifier for binary datasets.
    
    Source:
      Wang, Q., Luo, Z., Huang, J., Feng, Y., & Liu, Z. (2017).
      A Novel Ensemble Method for Imbalanced Data Learning: Bagging of Extrapolation‐SMOTE SVM.
      Computational intelligence and neuroscience, 2017(1), 1827016.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, random_state=None):
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.best_svm = None
        self.selected_kernel = None

    def __borderline_SMOTE(self, SV0, bootstrap_minority, bootstrap_majority):
        synthetic_samples = []
        set_bootstrap_minority = set(map(tuple, bootstrap_minority))
        set_SV0 = set(map(tuple, SV0))
        non_SV0 = np.array(list(set_bootstrap_minority - set_SV0))
        
        flag = 0
        if len(non_SV0) == 0:
            rng = check_random_state(self.random_state)
            random_indices = rng.choice(len(SV0), size=min(5, len(SV0)), replace=False)
            non_SV0 = SV0[random_indices, :-1]
            flag = 1

        # Handle 1D array case
        if non_SV0.ndim == 1 and len(non_SV0) > 0:
            non_SV0 = non_SV0.reshape(1, -1)
            
        # Create nearest neighbors model
        k = min(5, len(non_SV0))
        if k > 0:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(non_SV0)
        else:
            return np.array([])
        
        rng = check_random_state(self.random_state)
        n_synthetic = max(1, len(bootstrap_majority) - len(bootstrap_minority) - len(SV0))
        
        for r in range(n_synthetic):
            sv_idx = rng.randint(len(SV0))
            sv = SV0[sv_idx]
            xi = sv[:-1]
            
            if flag == 0:
                distances, indices = nbrs.kneighbors([xi])
                xi_t_idx = rng.choice(indices[0])
                xi_t = non_SV0[xi_t_idx]
            else:
                # Filter out xi if present
                mask = ~np.all(non_SV0 == xi, axis=1)
                if np.any(mask):
                    xi_t = non_SV0[mask][rng.choice(np.sum(mask))]
                else:
                    xi_t = non_SV0[rng.choice(len(non_SV0))]
            
            delta = rng.uniform(0, 1)
            y = self.best_svm.decision_function([xi])
            dist_to_hyperplane = abs(y) / np.linalg.norm(self.best_svm.coef_)
            
            if dist_to_hyperplane > 1e-6:
                x_new = xi + (delta * (xi - xi_t)) / dist_to_hyperplane
            else:
                x_new = xi + (delta * (xi - xi_t))
                
            synthetic_samples.append(x_new)
        
        return np.array(synthetic_samples)

    def __make_bootstraps(self, data, SV0_indices, minority_class):
        dc = {}
        rng = check_random_state(self.random_state)
        SV0 = data[SV0_indices]
        
        for i in range(self.n_estimators):
            data_no_supports = np.delete(data, SV0_indices, axis=0)
            bootstrap_indices = rng.choice(
                len(data_no_supports), 
                size=len(data_no_supports), 
                replace=True
            )
            bootstrap_data = data_no_supports[bootstrap_indices]
            
            # Get out-of-bag data
            in_bag_set = set(map(tuple, bootstrap_data))
            all_set = set(map(tuple, data_no_supports))
            oob_data = np.array(list(all_set - in_bag_set))
            
            # Combine all data
            union_d = np.concatenate((bootstrap_data, SV0), axis=0)
            
            # Prepare for SMOTE
            idx_min = np.where(bootstrap_data[:, -1] == minority_class)[0]
            bootstrap_minority = bootstrap_data[idx_min, :-1]
            idx_maj = np.where(bootstrap_data[:, -1] != minority_class)[0]
            bootstrap_majority = bootstrap_data[idx_maj, :-1]
            
            synthetic_samples_a = self.__borderline_SMOTE(
                SV0, bootstrap_minority, bootstrap_majority
            )
            
            if len(synthetic_samples_a) > 0:
                labels = np.full(synthetic_samples_a.shape[0], minority_class)
                synthetic_samples = np.column_stack((synthetic_samples_a, labels))
                union_d = np.concatenate((union_d, synthetic_samples), axis=0)
            
            dc[f'boot_{i}'] = {
                'boot': union_d, 
                'test': oob_data if len(oob_data) > 0 else None
            }
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("BEBS classifier is designed for binary classification only.")
            
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        
        training_data = np.column_stack((X, y))
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        # Perform initial grid search
        param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']}
        g_means_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
        stratified_kfold = StratifiedKFold(n_splits=2, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            svm.SVC(), 
            param_grid, 
            scoring=g_means_scorer, 
            cv=stratified_kfold, 
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.best_svm = grid_search.best_estimator_
        self.selected_kernel = grid_search.best_params_['kernel']
        support_indices = self.best_svm.support_
        support_labels = y[support_indices]
        
        # Select support vectors from minority class
        SV0_indices = support_indices[support_labels == minority_class]
        
        # Create bootstrap samples
        dcBoot = self.__make_bootstraps(training_data, SV0_indices, minority_class)
        
        # Train ensemble
        self.models = []
        for b in dcBoot:
            # Skip if no OOB data
            if dcBoot[b]['test'] is None or len(dcBoot[b]['test']) < 2:
                cls = svm.SVC(
                    kernel=self.selected_kernel, 
                    C=1.0, 
                    random_state=self.random_state
                )
            else:
                param_grid = {'C': [0.01, 0.1, 1, 10]}
                grid_search = GridSearchCV(
                    svm.SVC(kernel=self.selected_kernel, random_state=self.random_state),
                    param_grid,
                    cv=StratifiedKFold(n_splits=2, random_state=self.random_state),
                    n_jobs=-1
                )
                try:
                    grid_search.fit(
                        dcBoot[b]['test'][:, :-1], 
                        dcBoot[b]['test'][:, -1]
                    )
                    cls = grid_search.best_estimator_
                except:
                    cls = svm.SVC(
                        kernel=self.selected_kernel, 
                        C=1.0, 
                        random_state=self.random_state
                    )
            
            cls.fit(dcBoot[b]['boot'][:, :-1], dcBoot[b]['boot'][:, -1])
            self.models.append(cls)
        
        return self