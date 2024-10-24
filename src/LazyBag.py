"""
Description: This file contains a class that implements "Lazy Bagging (LazyBag)" algorithm using scikit-learn.
LazyBag is a multi-class classification algorithm

Source: Zhu, X. (2007, October). Lazy bagging for classifying imbalanced data. In Seventh IEEE International Conference on Data Mining (ICDM 2007) (pp. 763-768). IEEE.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
import math
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
#---------------------------------------------------------------    

class LazyBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(), n_neighbors = 5, beta = 0.99):
        self.n_estimator = n_estimator
        self.beta = beta
        self.models = []
        self.baseclassifier = estimator
        self.K = n_neighbors
        self.classes = []


    # Calculate the Information-gain Ratio (IR) for dataset attributes
    def __calculate_ir(self, dataset):
        data = dataset[:,:-1]
        entropy = []
        for col in range(data.shape[1]):
            unique_values, counts = np.unique(data[:, col], return_counts=True)
            probabilities = counts / np.sum(counts)
            entropy.append(-np.sum(probabilities * np.log2(probabilities)))

        # Compute the total entropy
        total_counts = np.sum(counts)
        total_probabilities = counts / total_counts
        total_entropy = -np.sum(total_probabilities * np.log2(total_probabilities))

        ir_values = [(total_entropy - e) / total_entropy for e in entropy]

        # Normalize IR values
        min_ir = min(ir_values)
        max_ir = max(ir_values)
        normalized_ir = [(ir - min_ir) / (max_ir - min_ir) for ir in ir_values]

        return normalized_ir


    #private function to make bootstrap samples
    def __make_bootstrapsamp_fit(self,test_instance, data, weights):
        classes, counts = np.unique(data[:,-1], return_counts=True)
        if len(classes) < 2:
            raise ValueError("Dataset should have at leat two data classes.")
        if min(counts) < self.K:
            self.K = min(counts)
        knn = KNeighborsClassifier(n_neighbors=self.K, algorithm='ball_tree', n_jobs=-1, metric='minkowski', metric_params={'w': np.array(weights)})
        knn.fit(data[:,:-1], data[:,-1])
        neighbors = knn.kneighbors([test_instance], return_distance=False)
        S = data[neighbors[0], :]
        cls = self.baseclassifier
        predict = []
        #loop through the required number of bootstraps
        for b in range(self.n_estimator):
            idx = [i for i in range(data.shape[0])]
            sidx   = np.random.choice(idx,replace=True,size=len(data)-self.K)
            B = data[sidx,:]  
            idx = [i for i in range(S.shape[0])]
            sidx   = np.random.choice(idx,replace=True,size=self.K)
            P = S[sidx, :]
            b_samp = np.vstack([B, P])
            model = clone(cls)
            #fit a decision tree classifier to the current sample
            model.fit(b_samp[:,:-1],b_samp[:,-1].reshape(-1, 1))
            #append the fitted model
            predict.append(model.predict([test_instance])[0].astype(int))

        return(predict)

          
    #predict from the ensemble
    def predict(self,X_train,y_train, X_test):
        self.classes = len(set(y_train))
        # Determin the value of K
        omega = math.log(len(X_train)**(1-self.beta), 4)
        K = round(len(X_train)*omega)
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        weights = self.__calculate_ir(training_data)
        predictions = np.zeros((len(X_test),self.n_estimator))
        i = 0
        for test in X_test:
            yp = self.__make_bootstrapsamp_fit(test, training_data, weights)
            yp = np.array(yp)
            #append predictions to storage list
            predictions[i,:] = yp
            i += 1
        #compute the ensemble prediction
        ypred, _ = stats.mode(predictions, axis=1, keepdims=False)
        ypred = np.array(ypred)
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


