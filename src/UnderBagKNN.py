"""
Description: This file contains a class that implements "Under-bagging KNN" algorithm using scikit-learn.
UnderBagKNN is a multi-class classification algorithm

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from scipy import stats
#---------------------------------------------------------------
  
class UnderBagKNN(object):
    #initializer
    def __init__(self,n_estimator=10, subsample_rate=1, n_neighbors = 5):
        self.n_estimator = n_estimator
        self.models     = []
        self.subsample_rate = subsample_rate
        self.n_neighbors = n_neighbors
        self.classes = []

        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):       
        classes, counts = np.unique(data[:,-1], return_counts=True)
        n_classes = len(classes)
        lowest_n = min(counts)
        if lowest_n < self.n_neighbors:
            print("Warning: the number of samples in the minority class is less than specified k-neighbor values "+\
                   "in the parameter. Therefore, it is changed to "+str(lowest_n))
            self.n_neighbors = lowest_n 
        dc   = {}
        #loop through the required number of bootstraps
        managed_data = np.empty((0,data.shape[1]))
        weights = np.array([])
        for c in classes:               
            n_resamples = round(self.subsample_rate*lowest_n*n_classes)
            idx = np.where(data[:,-1]==c)[0]
            class_data = data[idx,:]
            w=np.full(len(idx), (n_resamples)/(n_classes*len(idx)))
            weights=np.concatenate((weights, w))
            managed_data=np.concatenate((managed_data, class_data))
        for b in range(self.n_estimator):                        
            resampled = resample(managed_data, replace=True, n_samples=n_resamples, random_state=0, stratify=weights)
            b_samp = resampled

            #store results
            dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        self.classes = len(set(y_train))
        if self.classes < 2:
            raise ValueError("A dataset with the minimum of two class should be given.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data)
        #iterate through each bootstrap sample & fit a model ##
        cls = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        for b in dcBoot:
            #make a clone of the model
            model = clone(cls)
            #fit a decision tree classifier to the current sample
            model.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1])
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

    