"""
Description: This file contains a class that implements "Boundary Bagging (BBag)" algorithm using scikit-learn.
BBag was designed for multi-class datasets

Source: Boukir, S., & Feng, W. (2021, January). Boundary bagging to address training data issues in ensemble classification. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 9975-9981). IEEE.

Programmer: Yousef Abdi
Email: yousef.abdi@gmail.com
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy import stats
from collections import Counter
from sklearn.ensemble import BaggingClassifier
import random
#------------------------------------------------------------------------

class BBag(object):
    #initializer
    def __init__(self, n_estimator=10, estimator=DecisionTreeClassifier()):
        self.models     = []
        self.n_estimator = n_estimator
        self.baseclassifier = estimator
        self.classes = []

        
    #private function to make bootstrap samples
    def __make_bootstraps(self, data, counts, min_lbl, minority_count, margin):    
        dc   = {}    
        for i in range(self.n_estimator):
            idx = np.where(data[:,-1] == min_lbl)[0]
            minority_class = data[idx, :]
            sampled_data = minority_class
            for label, count in counts.items():
                if label != min_lbl:
                    class_indices = np.where(data[:,-1]==label)[0]
                    class_data = data[class_indices,:]
                    class_margins = margin[class_indices]
                    rem_samples = len(class_indices) - minority_count
                    rand_number = random.randint(int(random.uniform(0.2, 1)*rem_samples), rem_samples)
                    top_indices = np.argsort(class_margins)[-rand_number:]

                    mask = np.ones(len(class_margins), dtype=bool)
                    mask[top_indices] = False
                    new_class_data = class_data[mask]

                    sampled_data = np.concatenate((sampled_data, new_class_data), axis = 0)
            dc['boot_'+str(i)] = {'boot':sampled_data}     

        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        self.classes = len(set(y_train))
        if self.classes < 2:
            raise ValueError("The target class should have at least two classes of data.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        # finding minority and majority classes
        counts = Counter(training_data[:,-1])
        minority_count = 100000000
        majority_count = 0
        for label, count in counts.items():
            if count < minority_count:
                minority_count = count
                min_lbl = label
            if count > majority_count:
                majority_count = count
                max_lbl = label 
        # compute margin for each sample
        margin = np.zeros(len(training_data))
        y_pred = np.zeros(len(training_data))
        idx = 0
        init_ensemble = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=10,\
                                        random_state=0).fit(training_data[:,:-1], training_data[:,-1])
        for x in training_data:
            ensemple_pred = np.zeros(self.n_estimator)
            for i, estimator in enumerate(init_ensemble.estimators_):
                ensemple_pred[i] = estimator.predict(list([x[:-1]]))[0]
            votes = np.bincount(ensemple_pred.astype(int))
            last_elem = x[-1].astype(int)
            if last_elem < len(votes):
                v_y = votes[x[-1].astype(int)]
                other_votes = np.copy(votes)
                other_votes[x[-1].astype(int)] = 0
                v_c = np.max(other_votes)                
            else:
                v_y = 0
                v_c = np.max(votes)
            margin[idx] = (v_y - v_c) / (v_y + v_c)
            y_pred[idx] = init_ensemble.predict(list([x[:-1]]))[0]
            idx+=1
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data, counts, min_lbl, minority_count, margin)
        #iterate through each bootstrap sample & fit a model ##
        cls = self.baseclassifier
        for b in dcBoot:
            #make a clone of the model
            model = clone(cls)
            #fit a decision tree classifier to the current sample
            model.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1].reshape(-1, 1))
            #append the fitted model
            self.models.append(model)
          
    #predict from the ensemble
    def predict(self,X):
        #check we've fit the ensemble
        if not self.models:
            print('You must train the ensemble before making predictions!')
            return(None)
        #loop through each fitted model
        predictions = np.zeros((len(X), len(self.models)))
        i = 0
        for m in self.models:
            #make predictions on the input X
            yp = m.predict(X)
            #append predictions to storage list
            predictions[:,i]=yp
            i+=1
        #compute the ensemble prediction
        ypred, _ = stats.mode(predictions, axis=1, keepdims=False)
        ypred = np.array(ypred)
        #return the prediction
        return(ypred)
    #-------------------------------------------------------------------------------------------------------
    def predict_proba(self, X):
        n_classes = len(self.classes)
        probas = np.zeros((X.shape[0], n_classes))
        # Compute the class probabilities for each base estimator
        for estimator in self.models:
            probas += estimator.predict_proba(X)
        # Average the probabilities
        probas /= len(self.models)

        return probas