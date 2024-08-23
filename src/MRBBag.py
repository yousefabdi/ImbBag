"""
Description: This file contains a class that implements "Multi-class Roughly Balanced Bagging (MRBBag)" algorithm using scikit-learn.
MRBBag is a multi-class classification algorithm.
MRBBag has two modes: undersampling and oversampling

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
from scipy import stats
#---------------------------------------------------------------    

class MRBBag(object):
    #initializer
    def __init__(self, n_estimator=10, estimator=DecisionTreeClassifier(), sampling_method = "undersampling"):
        self.n_estimator = n_estimator
        self.sampling = sampling_method
        self.models     = []
        self.baseclassifier = estimator

        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):
        # finding minority and majority classes
        classes, counts = np.unique(data[:,-1], return_counts=True)
        lowest = min(counts)
        largest = max(counts)  
        probabilities = [1/len(classes) for i in range(len(classes))]
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        if self.sampling == "undersampling":            
            for b in range(self.n_estimator):
                while True:
                    multinomial_samples = np.random.multinomial(lowest, probabilities)
                    if not any(cnt == 0 for cnt in multinomial_samples):
                        break
                    lowest += 1            
                b_samp = np.empty((0, data.shape[1]))                                
                for j in range(len(classes)):
                    idx = np.where(data[:,-1]==classes[j])[0]
                    class_data = data[idx,:]
                    idx = [i for i in range(counts[j])]
                    sidx   = np.random.choice(idx,replace=True,size=multinomial_samples[j])
                    sampled_data = class_data [sidx, :]
                    b_samp = np.vstack([b_samp, sampled_data])
                dc['boot_'+str(b)] = {'boot':b_samp}
        else:            
            for b in range(self.n_estimator): 
                multinomial_samples = np.random.multinomial(largest, probabilities)           
                b_samp = np.empty((0, data.shape[1]))                
                for j in range(len(classes)):
                    idx = np.where(data[:,-1]==classes[j])[0]
                    class_data = data[idx,:]
                    idx = [i for i in range(counts[j])]
                    sidx   = np.random.choice(idx,replace=True,size=multinomial_samples[j])
                    sampled_data = class_data [sidx, :]
                    b_samp = np.vstack([b_samp, sampled_data])
                dc['boot_'+str(b)] = {'boot':b_samp}
            
            
        #return the bootstrap results
        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes < 2:
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
