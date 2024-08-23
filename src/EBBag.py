"""
Description: This file contains a class that implements "Exactly Balanced Bagging (EBBag)" algorithm using scikit-learn.
EBBag is a binary class classification algorithm.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import random
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
#---------------------------------------------------------------    

class EBBag(object):
    #initializer
    def __init__(self, estimator=DecisionTreeClassifier()):
        self.models     = []
        self.baseclassifier = estimator        

    def get_params(self, deep=True):
        return {"estimator": self.baseclassifier}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    #-------------------------------------------------------------------------------------------------

    #private function to make bootstrap samples
    def __make_bootstraps(self,data):    
        # finding minority and majority classes
        classes, counts = np.unique(data[:,-1], return_counts=True)
        if counts[0]>counts[1]:
            Lmaj = classes[0]
            Lmin = classes[1]
        else:
            Lmaj = classes[1]
            Lmin = classes[0]  
        Q = np.ceil(max(counts)/min(counts))   
        idx = np.where(data[:,-1]==Lmaj)[0]
        majority_data = data[idx,:]
        idx = np.where(data[:,-1]==Lmin)[0]
        minority_data = data[idx,:]
        majority_subsets = np.array_split(majority_data, Q)
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        for b in majority_subsets:
            bootstrap = np.concatenate((b, minority_data), axis=0)            
            #store results
            dc['boot_'+str(b)] = {'boot':bootstrap}
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes < 2:
            raise ValueError("A dataset with the minimum of two class should be given.")
        if unique_classes > 2:
            raise ValueError("EEBag is designed for binary class classification.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data)
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
        n_classes = 2
        probas = np.zeros((X.shape[0], n_classes))
        # Compute the class probabilities for each base estimator
        for estimator in self.models:
            probas += estimator.predict_proba(X)
        # Average the probabilities
        probas /= len(self.models)

        return probas
    