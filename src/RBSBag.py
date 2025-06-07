# RBSBag.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from collections import OrderedDict
import copy

from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class RBSBag(ImbalanceBaggingClassifier):
    """
    RBS Bagging classifier for multi-class classification.
    
    Based on:
      Huang, C., Huang, X., Fang, Y., Xu, J., Qu, Y., Zhai, P., ... & Li, J. (2020).
      Sample imbalance disease classification model based on association rule feature selection.
      Pattern Recognition Letters, 133, 280-286.
      
    This algorithm performs feature selection using an ARFS procedure before building
    an ensemble of base classifiers on balanced bootstrap samples. Voting is performed via a
    custom weighted voting scheme.
    
    Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=50, 
                 estimator=DecisionTreeClassifier(random_state=42), 
                 n_bins=10, threshold=0.01, 
                 min_support=0.05, min_estimator=2,
                 random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)
        self.n_bins = n_bins
        self.threshold = threshold
        self.min_support = min_support
        self.min_estimator = min_estimator
        self.selected_features_ = None
        self.models_weight_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params.update({
            "n_bins": self.n_bins,
            "threshold": self.threshold,
            "min_support": self.min_support,
            "min_estimator": self.min_estimator
        })
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        estimator_params = {}
        for key, value in parameters.items():
            if key in ["n_bins", "threshold", "min_support", "min_estimator"]:
                setattr(self, key, value)
            elif key.startswith("estimator__"):
                estimator_params[key.split('__', 1)[1]] = value
            else:
                setattr(self, key, value)
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        return self

    def __ARFS(self, data, classes, rng):
        selected_features = set()
        X_train, X_val, y_train, y_val = train_test_split(
            data[:, :-1], data[:, -1],
            test_size=0.2,
            stratify=data[:, -1],
            random_state=rng.randint(0, 1000000)
        )
        
        for class_label in classes:
            idx = np.where(y_train == class_label)[0]
            class_data = X_train[idx]
            
            # Discretize features
            df = pd.DataFrame(class_data)
            for col in df.columns:
                df[col] = pd.qcut(df[col], q=self.n_bins, duplicates='drop').astype(str)
            df = pd.get_dummies(df, columns=df.columns, drop_first=False)
            
            # Association rule mining
            frequent_itemsets = fpgrowth(df, min_support=self.min_support, use_colnames=True)
            if frequent_itemsets.empty:
                continue
                
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                    min_threshold=self.min_support)
            rules = rules.sort_values(by="confidence", ascending=False)
            
            # Feature selection
            features_ordered = []
            for _, row in rules.iterrows():
                features_ordered.extend(list(row["antecedents"]))
                features_ordered.extend(list(row["consequents"]))
            features_ordered = list(OrderedDict.fromkeys(features_ordered))
            
            clf = clone(self.estimator)
            accuracies = []
            current_features = []
            
            for feature in features_ordered:
                feature_idx = int(feature.split("_")[0])
                if feature_idx in current_features:
                    continue
                    
                current_features.append(feature_idx)
                clf.fit(X_train[:, current_features], y_train)
                y_pred = clf.predict(X_val[:, current_features])
                accuracy = f1_score(y_val, y_pred, average="weighted")
                
                if accuracies and accuracy <= accuracies[-1]:
                    current_features.remove(feature_idx)
                    break
                    
                accuracies.append(accuracy)
                
            selected_features.update(current_features)
            
        return sorted(selected_features)

    def __weighted_voting(self, X):
        check_is_fitted(self, ['models_', 'models_weight_'])
        predictions = np.array([model.predict(X) for model in self.models_])
        weights = np.array(self.models_weight_)
        
        # Weighted voting
        weighted_preds = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            weighted_preds[:, i] = np.sum(weights[:, np.newaxis] * (predictions == cls), axis=0)
            
        return self.classes_[np.argmax(weighted_preds, axis=1)]

    def fit(self, X, y):
        """
        Fit the RBSBag ensemble classifier.
        This procedure uses association rule feature selection (ARFS) to determine a subset
        of relevant features. Then, on the feature-augmented data, it iteratively builds an ensemble
        by generating bootstrap samples and assessing performance on out-of-bag (OOB) data via weighted F1-score.
        The process stops when a convergence criterion (change in F1 below a threshold) is met.
        """
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("At least two classes required")
            
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        training_data = np.column_stack((X, y))
        rng = check_random_state(self.random_state)
        
        # Feature selection
        self.selected_features_ = self.__ARFS(training_data, self.classes_, rng)
        if not self.selected_features_:
            self.selected_features_ = list(range(X.shape[1]))
            
        # Prepare feature-selected data
        s_f = copy.deepcopy(self.selected_features_) + [-1]
        featured_data = training_data[:, s_f]
        
        # Ensemble training
        self.models_ = []
        self.models_weight_ = []
        delta_f1 = []
        best_f1 = 0
        
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            resampled_data = np.vstack([
                resample(featured_data[featured_data[:, -1] == cls], 
                        replace=True, 
                        n_samples=max(1, int(0.8 * np.sum(featured_data[:, -1] == cls))),
                        random_state=rng.randint(0, 1000000))
                for cls in self.classes_
            ])
            
            # Train model
            model = clone(self.estimator)
            model.fit(resampled_data[:, :-1], resampled_data[:, -1])
            
            # OOB evaluation
            oob_mask = np.ones(len(featured_data), dtype=bool)
            oob_mask[np.concatenate([
                rng.choice(np.where(featured_data[:, -1] == cls)[0], 
                          size=np.sum(featured_data[:, -1] == cls) - np.sum(resampled_data[:, -1] == cls),
                          replace=False)
                for cls in self.classes_
            ])] = False
                
            oob_data = featured_data[oob_mask]
            if len(oob_data) > 0:
                y_pred = model.predict(oob_data[:, :-1])
                f1 = f1_score(oob_data[:, -1], y_pred, average="weighted")
                
                self.models_.append(model)
                self.models_weight_.append(f1)
                
                # Normalize weights
                total = sum(self.models_weight_)
                self.models_weight_ = [w/total for w in self.models_weight_]
                
                # Check convergence
                if len(self.models_) >= self.min_estimator:
                    ensemble_pred = self.__weighted_voting(oob_data[:, :-1])
                    current_f1 = f1_score(oob_data[:, -1], ensemble_pred, average="weighted")
                    delta_f1.append(abs(current_f1 - best_f1))
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        
                    if (len(delta_f1) > 5 and 
                        np.mean(delta_f1[-5:]) < self.threshold):
                        break
                        
        return self

    def predict(self, X):
        check_is_fitted(self, ['selected_features_', 'models_'])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
            
        featured_X = X[:, self.selected_features_]
        return self.__weighted_voting(featured_X)

    def predict_proba(self, X):
        raise NotImplementedError("Probability prediction not supported for RBSBag")