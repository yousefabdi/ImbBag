"""
Description: This file contains a class that implements "Bagging Ensemble Variation (BEV)" algorithm using scikit-learn.
BEV was designed for binary class datasets

Source: Li, C. (2007, March). Classifying imbalanced data using a bagging ensemble variation (BEV). In Proceedings of the 45th annual southeast regional conference (pp. 203-208). 

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import ADASYN
from scipy import stats
#------------------------------------------------------------------------

class BEV(object):
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
        # split majority and minority classes
        idx = np.where(data[:,-1]==Lmin)[0]
        d_minority = data[idx, :]    
        idx = np.where(data[:,-1]==Lmaj)[0]
        d_majority = data[idx, :]
        n_sets = np.ceil(len(d_majority)/len(d_minority))
        majority_sets = np.array_split(d_majority, n_sets)        
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        for i, set in enumerate(majority_sets):            
            b_samp = np.vstack([set, d_minority])
            #store results
            dc['boot_'+str(i)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes != 2:
            raise ValueError("BEV classifier is designed for binary classification only.")
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