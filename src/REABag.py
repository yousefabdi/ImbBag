"""
Description: This file contains a class that implements "resampling ensemble" algorithm using scikit-learn.
REABag is a multi-class classification algorithm

Source: Qian, Y., Liang, Y., Li, M., Feng, G., & Shi, X. (2014). A resampling ensemble algorithm for classification of imbalance problems. Neurocomputing, 143, 57-67.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from scipy import stats
#---------------------------------------------------------------    

class REABag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(), k = 2, alpha = 0, beta = 1):
        self.n_estimator = n_estimator
        self.k = k
        self.models = []
        self.alpha = alpha
        self.beta = beta
        self.baseclassifier = estimator
        self.classes = []

        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):
        classes, counts = np.unique(data[:,-1], return_counts=True)
        n_lowest = min(counts)
        n_largest = max(counts)  
        R = n_lowest / n_largest
        self.alpha = -0.097*R+1.428
        self.beta = 0.198*R+0.738
        S = np.empty(len(classes))
        dc   = {}        
        for i in range(len(classes)):
            t1 = (self.beta*n_largest-self.alpha*n_lowest)/(n_largest-n_lowest)
            t2 = ((self.alpha-self.beta)*n_largest*n_lowest)/((n_largest-n_lowest)*counts[i])
            S[i] = t1+t2
        for b in range(self.n_estimator):
            resampled_data = np.empty((0, data.shape[1]))
            resmp = []
            for i in range(len(classes)):
                resmp.append(round(counts[i]*S[i]))
                idx = np.where(data[:,-1] == classes[i])[0]
                class_data = data[idx, :]
                if resmp[-1]>counts[i]: # oversample
                    n_resample = resmp[-1] - counts[i]
                    for j in range(n_resample):
                        selected_index = np.random.randint(0, len(class_data))
                        selected_sample = class_data[selected_index,:]
                        knn = NearestNeighbors(n_neighbors=self.k)
                        knn.fit(class_data)
                        _, indices = knn.kneighbors([selected_sample])
                        k_nearest_neighbors = class_data[indices[0]]
                        random_weights = np.random.rand(self.k)
                        new_sample = np.average(k_nearest_neighbors, axis=0, weights=random_weights)
                        #new_sample = np.concatenate((new_sample, classes[i]), axis=1)
                        resampled_data = np.concatenate((resampled_data, [new_sample]), axis=0)
                else: # undersample
                    n_resample = counts[i] - resmp[-1]
                    for j in range(n_resample):
                        selected_index = np.random.randint(0, len(class_data))
                        class_data = np.delete(class_data, selected_index, axis=0)
                    resampled_data = np.concatenate((resampled_data, class_data), axis=0)
            dc['boot_'+str(b)] = {'boot':resampled_data}
            
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        self.classes = len(set(y_train))
        if self.classes < 2:
            raise ValueError("The dataset should has at lear two classes")
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
            model.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1].reshape(-1, 1).astype(int))
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
        n_classes = self.classes
        probas = np.zeros((X.shape[0], n_classes))
        # Compute the class probabilities for each base estimator
        for estimator in self.models:
            probas += estimator.predict_proba(X)
        # Average the probabilities
        probas /= len(self.models)

        return probas
