"""
Description: This file contains a class that implements "Bagging of Extrapolation‐SMOTE SVM (BEBS)" algorithm using scikit-learn.
BEBS was designed for binary class datasets

Source: Wang, Q., Luo, Z., Huang, J., Feng, Y., & Liu, Z. (2017). A Novel Ensemble Method for Imbalanced Data Learning: Bagging of Extrapolation‐SMOTE SVM. Computational intelligence and neuroscience, 2017(1), 1827016.

# Programmer: Yousef Abdi
Date: July 2024
License: MIT License
"""
#------------------------------------------------------------------------
import numpy as np
from sklearnex import patch_sklearn
from sklearn.base import clone
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from collections import Counter
from imblearn.metrics import geometric_mean_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn.model_selection import StratifiedKFold
import random
#------------------------------------------------------------------------

class BEBS(object):
    #initializer
    def __init__(self, n_estimator=10):
        self.best_svm = None
        self.selected_kernel = None
        self.n_estimator = n_estimator
        self.models = []

    def get_params(self, deep=True):
        return {"n_estimator": self.n_estimator}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    #----------------------------------------------------------------------------------------------

    def __borderline_SMOTE(self, SV0, bootstrap_minority,bootstrap_majority):
        synthetic_samples = []
        set_bootstrap_minority = set(map(tuple, bootstrap_minority))
        set_SV0 = set(map(tuple, SV0))
        non_SV0 = np.array(list(set_bootstrap_minority - set_SV0))
        flag = 0
        if((len(non_SV0)==0) or (np.ndim(non_SV0)==1 and len(non_SV0>0))):
            random_indices = random.sample(range(0,len(SV0)), random.randint(1, len(SV0))) 
            if len(random_indices)>1:
                non_SV0 = SV0[random_indices, :-1]
            else:
                tmp = SV0[random_indices, :-1]
                non_SV0 = tmp.reshape(1, -1)
            flag = 1

        if non_SV0.shape[0]>=6 and np.ndim(non_SV0)>1:
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(non_SV0)  # n_neighbors= 5
        elif non_SV0.shape[0]>1 and np.ndim(non_SV0)>1:
            nbrs = NearestNeighbors(n_neighbors=non_SV0.shape[0]-1, algorithm='ball_tree').fit(non_SV0) 
        elif non_SV0.shape[0]==1 and np.ndim(non_SV0)==2:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(non_SV0) 

        n_synthetic = len(bootstrap_majority)-len(bootstrap_minority)-len(SV0)
        if n_synthetic<1:
            n_synthetic = 1
        for r in range(n_synthetic):
            sv = SV0[np.random.choice(len(SV0))]
            xi = sv[:-1]
            distances, indices = nbrs.kneighbors([xi])
            if flag == 0:
                xi_t = non_SV0[np.random.choice(len(indices))]
            else:
                new_sv = non_SV0[~np.all(non_SV0 == xi, axis=1)]
                if new_sv.size != 0:
                    # Determine the number of rows in the new array
                    num_rows = new_sv.shape[0]
                    random_index = np.random.randint(0, num_rows)
                    xi_t = new_sv[random_index, :]
                else:
                    xi_t = non_SV0[np.random.choice(len(indices))]
            delta = np.random.uniform(0, 1)
            y = self.best_svm.decision_function([xi])
            dist_to_hyperplane = abs(y) / np.linalg.norm(self.best_svm.coef_)
            if (dist_to_hyperplane * np.linalg.norm(xi - xi_t))!=0:
                x_new = xi + (delta * (xi - xi_t)) / (dist_to_hyperplane * np.linalg.norm(xi - xi_t))
            else:
                x_new = xi + (delta * (xi - xi_t)) / dist_to_hyperplane
            synthetic_samples.append(x_new)

        synthetic_samples = np.array(synthetic_samples)
        return synthetic_samples

    #private function to make bootstrap samples
    def __make_bootstraps(self, data, SV0_indices, minority_class):    
        dc   = {}  
        idx = np.where(data[:,-1]==minority_class)[0]
        SV0 = data[SV0_indices]
        for i in range(self.n_estimator):
            data_no_supports = np.delete(data, SV0_indices, axis=0)
            bootstrap_indices = np.random.choice(len(data_no_supports), size=len(data_no_supports), replace=True)
            bootstrap_data = data_no_supports[bootstrap_indices]
            # Identify the out-of-bag instances
            set_data_no_supports = set(map(tuple, data_no_supports))
            set_bootstrap_data = set(map(tuple, bootstrap_data))
            oob_data = np.array(list(set_data_no_supports - set_bootstrap_data))
            unique_values, counts = np.unique(oob_data[:,-1], return_counts=True)
            union_d = np.concatenate((bootstrap_data, SV0), axis=0)
            idx = np.where(bootstrap_data[:,-1]==minority_class)[0]
            bootstrap_minority = bootstrap_data[idx,:-1]
            # Add minority class to the out of bag-of-bag if no minority samples were selected
            indx = np.where(unique_values==minority_class)[0]
            if len(unique_values)<2 or counts[indx]<3:
                idx_ = np.where(data[:,-1]==minority_class)[0]
                oob_data = np.concatenate((oob_data, data[idx_,:]), axis=0)
            idx_m = np.where(bootstrap_data[:,-1]!=minority_class)[0]
            bootstrap_majority = bootstrap_data[idx_m,:-1]
            synthetic_samplesa = self.__borderline_SMOTE(SV0, bootstrap_minority, bootstrap_majority)
            labels = np.array([minority_class for row in synthetic_samplesa]).astype(int)
            synthetic_samples = np.column_stack((synthetic_samplesa, labels))
            union_d = np.concatenate((union_d, synthetic_samples), axis = 0)
            #sampled_data = np.concatenate((sampled_data, new_class_data), axis = 0)
            #dc['boot_'+str(i)] = {'boot':sampled_data}     
            dc['boot_'+str(i)] = {'boot':union_d,'test':oob_data}

        return(dc)
    
    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes > 2:
            raise ValueError("BEBS classifier is designed for binary classification only.")
        #package the input data
        training_data = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        # finding minority and majority classes
        class_counts = Counter(training_data[:,-1])
        minority_class = int(min(class_counts, key=class_counts.get))
        majority_class = int(max(class_counts, key=class_counts.get))
        
        # Define the parameter grid for the SVM
        param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']}  # 
        # Define G-means as the scoring metric
        g_means_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
        stratified_kfold = StratifiedKFold(n_splits=2)
        patch_sklearn()
        grid_search = GridSearchCV(svm.SVC(), param_grid, scoring=g_means_scorer, cv=stratified_kfold, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.best_svm = grid_search.best_estimator_
        best_params = grid_search.best_params_
        # Get the selected kernel function
        self.selected_kernel = best_params['kernel']
        support_indices = self.best_svm.support_
        support_labels = y_train[support_indices]
        # Identify the support vectors that belong to the minority class
        SV0_indices = support_indices[support_labels == minority_class]        
        #make bootstrap samples
        dcBoot = self.__make_bootstraps(training_data, SV0_indices, minority_class)
        #iterate through each bootstrap sample & fit a model ##
        
        cnt = -1
        for b in dcBoot:
            cnt = cnt+1
            #make a clone of the model
            param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']} # 0.001, 0.01, 0.1, 1, 10            
            patch_sklearn()
            svc = svm.SVC(kernel=self.selected_kernel, probability=False)
            stratified_kfold = StratifiedKFold(n_splits=2)
            grid_search = GridSearchCV(svc, param_grid, cv=stratified_kfold, error_score='raise', n_jobs=-1)
            uniq, counts = np.unique(dcBoot[b]['test'][:,-1], return_counts=True)
            if np.any(counts<2) or len(counts)<2:
                cls = svm.SVC(kernel=self.selected_kernel, C=1)
            else:
                grid_search.fit(dcBoot[b]['test'][:,:-1], dcBoot[b]['test'][:,-1])
                cls = svm.SVC(kernel=self.selected_kernel, C=grid_search.best_params_['C'])
            
            #fit a decision tree classifier to the current sample
            cls.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1])
            #append the fitted model
            self.models.append(cls)
          
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