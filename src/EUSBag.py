"""
Description: This file contains a class that implements "Evolutionary under-sampling based bagging ensemble (EUSBag)" algorithm using scikit-learn.
EUSBag is a multi-class classification algorithm

Source: Sun, B., Chen, H., Wang, J., & Xie, H. (2018). Evolutionary under-sampling based bagging ensemble method for imbalanced data classification. Frontiers of Computer Science, 12, 331-350.

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
import pygad
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
from scipy import stats
#---------------------------------------------------------------    

class EUSBag(object):
    #initializer
    def __init__(self,n_estimator=10, estimator=DecisionTreeClassifier(),population_size=50, \
                  num_generations = 25, alpha = 0.5, beta = 0.5):
        self.n_estimator = n_estimator
        self.population_size = population_size
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        self.best_fitness = None
        self.best_sol = []
        self.models     = []
        self.majority_data = None
        self.minority_data = None
        self.Lmaj = None
        self.Lmin = None
        self.baseclassifier = estimator
        if self.beta + self.beta !=1:
            raise ValueError("alpha and beta parameters should sum up to 1.")
    
    def get_params(self, deep=True):
        return {"n_estimator": self.n_estimator, "estimator": self.baseclassifier, "population_size": self.population_size, \
                "num_generations": self.num_generations, "alpha": self.alpha, "beta": self.beta}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    #-------------------------------------------------------------------------------------------------

    # Define the fitness function
    def __fitness_func(self, ga_instance, solution, solution_idx):
        clf = self.baseclassifier
        gmean_list = []
        id = np.where(solution == 1)[0]
        X = self.majority_data[id, :-1]
        X = np.concatenate((X, self.minority_data[:,:-1]),axis=0)  
        y = self.majority_data[id, -1]
        y = np.concatenate((y, self.minority_data[:,-1]),axis=0)
        _, counts = np.unique(y, return_counts=True)
        if min(counts)<5 and min(counts)>1:
            skf = StratifiedKFold(n_splits=min(counts))
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                gmean = geometric_mean_score(y_test, y_pred)
                gmean_list.append(gmean)
        elif min(counts)==1:
            clf.fit(X, y)
            y_pred = clf.predict(X)
            gmean = geometric_mean_score(y, y_pred)
            gmean_list.append(gmean)
        else:
            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                gmean = geometric_mean_score(y_test, y_pred)
                gmean_list.append(gmean)
        avg_gmean = np.mean(gmean_list)

        Aver_Q = []
        for i in range(self.population_size):
            other_sol = ga_instance.population[i]
            if i != solution_idx:
                N00 = np.sum((solution == 0) & (other_sol == 0))
                N11 = np.sum((solution == 1) & (other_sol == 1))
                N01 = np.sum((solution == 0) & (other_sol == 1))
                N10 = np.sum((solution == 1) & (other_sol == 0))   
                with np.errstate(divide='ignore', invalid='ignore'): 
                    val = (N11*N00-N10*N01)/(N11*N00+N10*N01)            
                    if not (np.isnan(val) or np.isinf(val)):                        
                        Aver_Q.append(val)
                    else:
                        Aver_Q.append(0)
        
        first_term = np.average(avg_gmean)
        with np.errstate(divide='ignore', invalid='ignore'):
            second_term = self.alpha*abs((1-len(self.minority_data)/sum(solution)))
        if np.isnan(second_term) or np.isinf(second_term):
            second_term = 0
        third_term = self.beta*np.average(Aver_Q)
        Fitness = first_term  - second_term - third_term

        return Fitness
    #-------------------------------------------------------------------------------------------------------    
   
    def __on_generation(self, ga_instance):
        global population
        population = ga_instance.population
        current_solution, current_fitness, _ = ga_instance.best_solution()
         
        #if self.best_fitness is not None and current_fitness <= self.best_fitness:
            #return True
        # Update the best fitness.
        self.best_fitness = current_fitness
        self.best_sol = current_solution
        print(f"The best fitness value: {self.best_fitness}")
        #return False
    #-------------------------------------------------------------------------------------------------------

    #private function to make bootstrap samples
    def __make_evolutionary_bootstrap(self,d_majority, d_minority):
        #initialize output dictionary & unique value count
        dc   = {}
        # Evolutionary settings
        self.best_fitness = None
        num_generations = self.num_generations        
        num1 = round(0.2*self.population_size)
                            
        self.majority_data = d_majority
        self.minority_data = d_minority
        #loop through the required number of bootstraps
        for b in range(self.n_estimator):
            ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating= int(0.65 * self.population_size),
                       fitness_func=self.__fitness_func,
                       sol_per_pop=self.population_size,
                       num_genes=len(d_majority),
                       gene_type=int, init_range_low=0, init_range_high=2,
                       mutation_by_replacement=True, parent_selection_type="rws",
                       mutation_percent_genes=5, crossover_probability=0.9,  
                       mutation_probability=0.1, on_generation=self.__on_generation,
                       crossover_type="two_points", keep_parents=num1)
            print(f"Evolutionary selection for the base learner: {b+1}")
            ga_instance.run()   
            solution, _, _ = ga_instance.best_solution()
            idx = np.where(solution == 1)[0]
            d_majority_sampled = d_majority[idx, :]
            b_samp = np.vstack([d_majority_sampled, d_minority])

            #store results
            dc['boot_'+str(b)] = {'boot':b_samp}
        #return the bootstrap results
        return(dc)
    #-------------------------------------------------------------------------------------------------------

    #train the ensemble
    def fit(self,X_train,y_train):
        unique_classes = len(set(y_train))
        if unique_classes != 2:
            raise ValueError("EUSBag classifier is designed for binary classification only.")
                # finding minority and majority classes
        classes, counts = np.unique(y_train, return_counts=True)
        if counts[0]>counts[1]:
            self.Lmaj = classes[0]
            self.Lmin = classes[1]
        else:
            self.Lmaj = classes[1]
            self.Lmin = classes[0]   
        # split majority and minority classes        
        idx = np.where(y_train==self.Lmin)[0]
        d_minority =np.concatenate((X_train[idx, :],y_train[idx].reshape(-1,1)),axis=1)    
        idx = np.where(y_train==self.Lmaj)[0]
        d_majority = np.concatenate((X_train[idx, :],y_train[idx].reshape(-1,1)),axis=1)   
    
        #iterate through each bootstrap sample & fit a model ##
        dcBoot = self.__make_evolutionary_bootstrap(d_majority, d_minority)
        cls = self.baseclassifier
        for b in dcBoot:
            #make a clone of the model
            model = clone(cls)
            #make bootstrap samples
            #fit a decision tree classifier to the current sample
            model.fit(dcBoot[b]['boot'][:,:-1],dcBoot[b]['boot'][:,-1].reshape(-1, 1))
            #append the fitted model
            self.models.append(model)
    #-------------------------------------------------------------------------------------------------------

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