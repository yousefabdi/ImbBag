"""
Description: This file contains a class that implements "Cost-sensitive Bagging (CostBag)" algorithm using scikit-learn.
CostBag is a multi-class classification algorithm.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from scipy import stats
#------------------------------------------------------------------------

class CostBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced')):
        self.n_estimator = n_estimator
        self.models     = []
        self.baseclassifier = estimator
        self.classes = []

        
    #private function to make bootstrap samples
    def __make_bootstraps(self,data):
        #initialize output dictionary & unique value count
        dc   = {}
        #loop through the required number of bootstraps
        for b in range(self.n_estimator):
            X_class_resampled, y_class_resampled  = resample(data[:,:-1], data[:,-1], \
                            replace=True, stratify=data[:,-1])
            
            b_samp = np.concatenate((X_class_resampled,y_class_resampled.reshape(-1,1)),axis=1)
            #store results
            dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    

    #train the ensemble
    def fit(self,X_train,y_train,print_metrics=False):
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        self.classes = len(set(y_train))
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data)
        #initialise metric arrays
        accs = np.array([])
        pres = np.array([])
        recs = np.array([])
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
        n_classes = self.classes
        probas = np.zeros((X.shape[0], n_classes))
        # Compute the class probabilities for each base estimator
        for estimator in self.models:
            probas += estimator.predict_proba(X)
        # Average the probabilities
        probas /= len(self.models)

        return probas
