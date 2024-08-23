"""
Description: This file contains a class that implements "Roughly Balanaced Bagging (RBBag)" algorithm using scikit-learn.
RBBag is a multi-class classification algorithm

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
#----------------------------------------------------------
    
class RBBag(object):
    #initializer
    def __init__(self, n_estimator=50, estimator=DecisionTreeClassifier()):
        self.n_estimator = n_estimator
        self.models     = []
        self.baseclassifier = estimator

    def get_params(self, deep=True):
        return {"n_estimator": self.n_estimator, "estimator": self.baseclassifier}
    
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
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        for b in range(self.n_estimator):
            idx = [i for i in range(d_majority.shape[0])]
            size = np.random.negative_binomial(len(d_minority), 0.5, size=1)
            sidx   = np.random.choice(idx,replace=True,size=size)
            d_majority_sampled = d_majority[sidx,:]  

            idx = [i for i in range(d_minority.shape[0])]
            sidx   = np.random.choice(idx,replace=True,size=len(idx))
            d_minority_sampled = d_minority[sidx,:]  
            
            b_samp = np.vstack([d_majority_sampled, d_minority_sampled])

            #store results
            dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes != 2:
            raise ValueError("RBBag classifier is designed for binary classification only.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data)
        #iterate through each bootstrap sample & fit a model ##
        cls = self.baseclassifier
        for b in dcBoot:
            #make a clone of the model
            model = clone(cls)
            #fit a ‌base classifier to the current sample
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
