"""
Description: This file contains a class that implements "RBS Bagging" algorithm using scikit-learn.
RBSBag is a multi-class classification algorithm

Source: Huang, C., Huang, X., Fang, Y., Xu, J., Qu, Y., Zhai, P., ... & Li, J. (2020). Sample imbalance disease classification model based on association rule feature selection. Pattern Recognition Letters, 133, 280-286.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import random
from collections import Counter
from scipy import stats
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import resample
import copy
from collections import OrderedDict
#------------------------------------------------------------------------

class RBSBag(object):
    #initializer
    def __init__(self, estimator = DecisionTreeClassifier(), n_bins = 10, threshold = 0.01, min_support = 0.05, min_estimator = 2):
        self.selected_features = None
        self.baseclassifier = estimator
        self.models = []
        self.models_weight = []
        self.classes = []
        self.threshold = threshold
        self.min_support = min_support
        self.min_estimator = min_estimator
        self.n_bins = n_bins

    # ARFS algorithm
    def __ARFS(self, data, classes):
        whole_selected_features = []
        X_train, X_val, y_train, y_val = train_test_split(data[:,:-1], data[:,-1], \
                             test_size=0.2, stratify=data[:,-1])
        for class_label in classes.keys():
            idx = np.where(y_train==class_label)[0]
            class_data = X_train[idx, :]

            df = pd.DataFrame(class_data)
            for col in df.columns:
                df[col] = pd.qcut(df[col], q=self.n_bins, duplicates='drop').astype(str)
            df = pd.get_dummies(df, columns=df.columns, drop_first=False)
            frequent_itemsets = fpgrowth(df, min_support=self.min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_support)
            rules = rules.sort_values(by='confidence', ascending=False)
            features_ordered = []
            for _, row in rules.iterrows():
                features_ordered.extend(list(row['antecedents']))
                features_ordered.extend(list(row['consequents']))
            features_ordered = list(OrderedDict.fromkeys(features_ordered))

            clf = self.baseclassifier
            accuracies = []
            selected_features = []

            for feature in features_ordered:
                value_before_underscore = feature.split('_')[0]
                if int(value_before_underscore) in selected_features:
                    continue
                selected_features.append(int(value_before_underscore))
                selected_features = list(set(selected_features))
                clf.fit(X_train[:,selected_features], y_train)
                y_pred = clf.predict(X_val[:,selected_features])
                accuracy = accuracy_score(y_val, y_pred)
                accuracies.append(accuracy)
                if len(accuracies) > 1 and accuracies[-1] <= accuracies[-2]:
                    selected_features.remove(int(value_before_underscore))
                    accuracies.pop()
                    break
            whole_selected_features.append(selected_features)   
        whole_selected_features = list(set(x for sublist in whole_selected_features for x in sublist))
        return whole_selected_features          

    #private function to make bootstrap samples
    def __make_bootstraps(self, data, classes, minority_class):    
        resampled_data = np.empty((0, data.shape[1]))
        oob_data = np.empty((0, data.shape[1]))
        for class_label in classes.keys():
            idx = np.where(data[:,-1]==class_label)[0]
            idx_minor = np.where(data[:,-1]==minority_class)[0]
            class_data = data[idx,:]
            sel_proba = random.uniform(len(idx_minor)/(len(idx)*3), len(idx_minor)/len(idx))
            n_samples = int(len(idx) * sel_proba)
            resampled_indices = resample(idx, replace=True, n_samples=n_samples, random_state=42)
            resampled_class = data[resampled_indices,:]
            resampled_data = np.concatenate((resampled_data, resampled_class), axis=0)

            oob_indices = list(set(idx) - set(resampled_indices))
            oob_data = data[oob_indices,:]       

        return resampled_data, oob_data
    
    # Weighted Majority Voting Prediction
    def __weighted_voting(self, state, models, weights, X):
        if state == 1:
            predictions = [model.predict(X[:,:-1]) for model in models]
        else:
            predictions = [model.predict(X) for model in models]
        final_predictions = []
        for i in range(X.shape[0]):
            weighted_votes = {}
            for j in range(len(models)):
                key = int(predictions[j][i])
                weighted_votes[key] = weighted_votes.get(key, 0) + weights[j]
            final_prediction = max(weighted_votes, key=weighted_votes.get)
            final_predictions.append(final_prediction)
        final_predictions = np.array(final_predictions).astype(int)    
        return final_predictions

    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes < 2:
            raise ValueError("At least two classes of data should be exist in dataset.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        # finding minority and majority classes
        classes = Counter(training_data[:,-1])
        minority_class = min(classes, key=classes.get)
        majority_class = max(classes, key=classes.get)
        self.classes = classes.keys()
        self.selected_features = self.__ARFS(training_data, classes)
        s_f = copy.deepcopy(self.selected_features)
        s_f.append(-1)
        featured_data = training_data[:, s_f]
        delta_f1 = [] 
        f1 = []   
        while(True):            
            resampled_data, oob_data = self.__make_bootstraps(featured_data, classes, minority_class)
            #iterate through each bootstrap sample & fit a model ##
            cls = clone(self.baseclassifier)
            cls.fit(resampled_data[:,:-1], resampled_data[:,-1])            
            y_pred = cls.predict(oob_data[:,:-1])
            temp_f1 = f1_score(oob_data[:,-1], y_pred, average='weighted')
            if temp_f1>0:
                self.models.append(cls)
                f1.append(temp_f1)
                self.models_weight.append(f1[-1])
                total = sum(self.models_weight)
                self.models_weight = [w/total for w in self.models_weight]
                best_f1 = max(f1)
                ensemble_pred = self.__weighted_voting(1, self.models, self.models_weight, oob_data)
                if len(self.models_weight)>2:                
                    f_val = f1_score(oob_data[:,-1], ensemble_pred, average='weighted')
                    delta_f1.append(f_val)
                    if len(delta_f1) > self.min_estimator and abs(delta_f1[-1] - best_f1) < self.threshold:
                        break  

    #predict from the ensemble
    def predict(self,X):
        features = self.selected_features
        featured_X = X[:, features]
        #check we've fit the ensemble
        if not self.models:
            print('You must train the ensemble before making predictions!')
            return(None)
        #compute the ensemble prediction
        ypred = self.__weighted_voting(0, self.models, self.models_weight, featured_X)
        #return the prediction
        return(ypred)
    #-------------------------------------------------------------------------------------------------------
    
    def predict_proba(self, X):
        n_classes = 2
        probas = np.zeros((X.shape[0], n_classes))
        # Compute the class probabilities for each base estimator
        for estimator in self.models:
            probas += estimator.predict_proba(X)
        # Average the probabilities
        probas /= len(self.models)

        return probas
