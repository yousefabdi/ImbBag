"""
Description: This file contains a class that implements "Random Balance ensembles for multiclass imbalance learning (MultiRandBalBag)" algorithm using scikit-learn.
MultiRandBalBag is a multi-class classification algorithm

Source: Rodríguez, J. J., Diez-Pastor, J. F., Arnaiz-Gonzalez, A., & Kuncheva, L. I. (2020). Random balance ensembles for multiclass imbalance learning. Knowledge-Based Systems, 193, 105434.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
import math
from scipy import stats
#------------------------------------------------------------------------

class MultiRandBalBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(), k_neighbors=5):
        self.n_estimator = n_estimator
        self.K = k_neighbors
        self.models     = []
        self.baseclassifier = estimator

    # Function that Implements SMOTE for Just One Target Class    
    def __custom_smote(self, data, count, k_neighbors):
        nbrs = NearestNeighbors(algorithm='auto', n_neighbors=k_neighbors).fit(data)
        neighbours = nbrs.kneighbors(data, return_distance=False)[:, 1:]
        new_samples = []
        while len(new_samples) < count:
            sample = data[np.random.randint(len(data))]
            neighbour = data[np.random.choice(neighbours[np.where(data == sample)[0][0]])]
            new_samples.append(sample + np.random.random() * (neighbour - sample))
        return new_samples


    #private function to make bootstrap samples
    def __random_balance_bootstraps(self,training_data):
        classes = np.unique(training_data[:,-1])
        w = []
        w = np.random.rand(len(classes))
        s_w = sum(w)
        X = np.empty((0,training_data.shape[1]))
        for i in range(len(classes)):
            idx = np.where(training_data[:,-1] == classes[i])[0]
            ni = max(math.floor(training_data.shape[0]*w[i]/s_w),2) 
            data = training_data[idx,:]           
            if ni <= len(idx):                                
                resamples = data[np.random.choice(data.shape[0], ni, replace=False), :]
                X = np.concatenate((resamples, X), axis=0)
            else:                
                data_res = self.__custom_smote(data, count=ni-len(idx), k_neighbors=self.K)
                X = np.concatenate((X, np.array(data_res)), axis = 0)            
        return X
    

    #train the ensemble
    def fit(self,X_train,y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        if min(counts)<self.K:
            self.K = min(counts)
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)        
        for i in range(self.n_estimator):
            dcBoot = self.__random_balance_bootstraps(training_data)
            cls = self.baseclassifier
            #make a clone of the model
            model = clone(cls)
            #fit a base classifier to the current sample
            model.fit(dcBoot[:,:-1],dcBoot[:,-1].reshape(-1, 1))
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
    
#-------------------------------------------------------------------------------------
