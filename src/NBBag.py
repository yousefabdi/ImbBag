"""
Description: This file contains a class that implements "Neighborhood Balanced Bagging Classifier (NBBag)" algorithm using scikit-learn.
NBBag is a binary class classification algorithm

Source: Błaszczyński, J., & Stefanowski, J. (2015). Neighbourhood sampling in bagging for imbalanced data. Neurocomputing, 150, 529-542.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
import random
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from Metrics import HVDM
from Metrics import HVDMstd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
#---------------------------------------------------------------
__author__ = "Yousef Abdi"
__email__ = "yousef.abdi@gmail.com"
#---------------------------------------------------------------    
class NBBag(object):

    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(), n_neighbors=5, phi = None, metric="hvdm", sampling_method="undersampling"):
        self.n_estimator = n_estimator
        self.sampling_method = sampling_method
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.phi = phi
        if phi == None and sampling_method == "undersampling":
            self.phi = 0.5
        elif phi == None and sampling_method == "oversampling":
            self.phi = 2
        self.models     = []
        self.baseclassifier = estimator

    def get_params(self, deep=True):
        return {"n_estimator": self.n_estimator, "estimator": self.baseclassifier, "n_neighbors": self.n_neighbors, \
                "phi": self.phi, "metric":self.metric, "sampling_method": self.sampling_method}
    
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
            Nmaj = counts[0]
            Lmaj = classes[0]
            Nmin = counts[1]
            Lmin = classes[1]
        else:
            Nmaj = counts[1]
            Lmaj = classes[1]
            Nmin = counts[0]
            Lmin = classes[0]
        #specifying the number of sampling according to the sampling method
        if self.sampling_method == "undersampling":
            n_samples = 2*Nmin
        else:
            n_samples = Nmin + Nmaj
        # -- setting weights to the training samples ---
        if self.metric == "hvdm":    
            hvdm_ = HVDM() # set metric HVDM    -  This can be used: 
        elif self.metric == "hvdmstd":
            hvdm_ = HVDMstd(data[:,-1])
        # Create the k-NN classifier
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=hvdm_.calculate_hvdm, n_jobs = -1)
        # Fit the classifier to the data
        knn.fit(data[:,:-1])
        # Compute the k-nearest neighbors for each sample
        neighbors = knn.kneighbors(data[:,:-1], return_distance=False)
        # Weighting according to the NBBag 
        weights = np.empty(data.shape[0])
        maj_idx= np.where(data[:,-1]==Lmaj)[0]
        weights[maj_idx] = (Nmin/Nmaj)*0.5
        min_idx= np.where(data[:,-1]==Lmin)[0]
        for i in min_idx:
            nbs = neighbors[i]
            lbls = data[nbs,-1]
            classes, counts = np.unique(lbls, return_counts=True)
            c = dict(zip(classes, counts))
            if len(classes) > 1:
                L = (c[Lmaj]**self.phi)/self.n_neighbors
                weights[i] = 0.5*(L+1)
            elif len(classes) == 1 and int(classes[0]) == int(Lmaj):
                L = (counts[0]**self.phi)/self.n_neighbors
                weights[i] = 0.5*(L+1)
            else:
                weights[i] = 0.5*(1/self.n_neighbors + 1)
        
        #initialize output dictionary & unique value count
        dc   = {}
        #get sample size
        b_size = data.shape[0]
        #get list of row indexes
        idx = [i for i in range(b_size)]
        #loop through the required number of bootstraps
        if self.sampling_method == "undersampling":
            for b in range(self.n_estimator):
                #obtain boostrap samples with replacement
                b_samp = np.array(random.choices(data, weights=weights, k=n_samples))
                #store results
                dc['boot_'+str(b)] = {'boot':b_samp}
        else:
            y = data[:,-1] 
            maj_idx = np.where(y == Lmaj)[0]           
            d_majority = data[maj_idx,:]
            min_idx = np.where(y == Lmin)[0]    
            d_minority = data[min_idx,:]
            for b in range(self.n_estimator):
                n_samples_majority = int(np.sum(weights[maj_idx]))
                n_samples_minority = int(np.sum(weights[min_idx]))    
                
                norm_weights = weights[min_idx]/sum(weights[min_idx])
                indices_upsampled = np.random.choice(len(d_minority), size=n_samples_minority, \
                            replace=True, p=norm_weights)
                d_minority_upsampled = d_minority[indices_upsampled,:]

                norm_weights = weights[maj_idx]/sum(weights[maj_idx])
                indices_upsampled_majority = np.random.choice(len(d_majority), \
                        size=n_samples_majority, replace=True, p=norm_weights)
                d_majority_upsampled = d_majority[indices_upsampled_majority,:]
  
                b_samp = np.vstack([d_majority_upsampled, d_minority_upsampled])
                #store results
                dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    

    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes > 2:
            raise ValueError("BinaryBagging is designed for binary classification only.")
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
