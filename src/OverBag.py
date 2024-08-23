"""
Description: This file contains a class that implements "OverBagging (OverBag)" algorithm using scikit-learn.
OverBag is a binary class classification algorithm

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import random
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy import stats
#---------------------------------------------------------------
__author__ = "Yousef Abdi"
__email__ = "yousef.abdi@gmail.com"
__version__ = "0.1"
#---------------------------------------------------------------    
class OverBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier()):
        self.n_estimator = n_estimator
        self.models     = []
        self.baseclassifier = estimator

        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):        
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        for b in range(self.n_estimator):
            resamp = RandomOverSampler(sampling_strategy='auto', random_state=random.randint(1,100))
            X_resampled, y_resampled = resamp.fit_resample(data[:,:-1], data[:, -1])
            b_samp = np.concatenate((X_resampled,y_resampled.reshape(-1,1)),axis=1)

            #store results
            dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes < 2:
            raise ValueError("A dataset with the minimum of two class should be given")
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

    