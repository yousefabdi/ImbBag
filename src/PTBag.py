"""
Description: This file contains a class that implements "PT-Bagging" algorithm using scikit-learn.
PT-Bagging is a binary class classification algorithm

Source: Collell, G., Prelec, D., & Patil, K. R. (2018). A simple plug-in bagging ensemble based on threshold-moving for classifying binary and multiclass imbalanced data. Neurocomputing, 275, 330-340.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from scipy import stats
#---------------------------------------------------------------    
class PTBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier()):
        self.n_estimator = n_estimator        
        self.lamda_k = []
        self.models  = []
        self.class_labels = []
        self.ensemble = BaggingClassifier(estimator=estimator, n_estimators=n_estimator, random_state=42)
        

    #train the ensemble
    def fit(self, X_train, y_train):
        unique, counts = np.unique(y_train, return_counts=True)
        sorted_indices = np.argsort(unique)
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]
        self.class_labels = unique
        class_counts = dict(zip(unique, counts))
        total_samples = len(y_train)
        self.lamda_k = {k: v / total_samples for k, v in class_counts.items()}

        self.ensemble.fit(X_train, y_train)
            
    #predict from the ensemble
    def predict(self,X):
        cls_proba=np.array([])
        spl_count = 0
        for x in X:
            proba = np.empty((self.n_estimator, len(self.class_labels)))
            i=0
            for estimator in self.ensemble.estimators_:
                proba[i,:] = estimator.predict_proba([x])
                i+=1
            avg_proba = np.sum(proba, axis = 0)/self.n_estimator
            for c in range(len(self.class_labels)):
                avg_proba[c] = avg_proba[c]/self.lamda_k[self.class_labels[c]]

            ind = np.argmax(avg_proba)
            cls_proba = np.append(cls_proba,self.class_labels[ind])

        # Make predictions
        return cls_proba
