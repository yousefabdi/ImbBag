import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_random_state
from imblearn.metrics import geometric_mean_score
import pygad
import warnings

from ImbalanceBaggingClassifier import ImbalanceBaggingClassifier

class EUSBag(ImbalanceBaggingClassifier):
    """
    Description: This file contains a class that implements "Evolutionary under-sampling based bagging ensemble (EUSBag)" algorithm using scikit-learn.
    EUSBag is a multi-class classification algorithm

    Source: Sun, B., Chen, H., Wang, J., & Xie, H. (2018). Evolutionary under-sampling based bagging ensemble method for imbalanced data classification. Frontiers of Computer Science, 12, 331-350.

    # Programmer: Yousef Abdi
    Date: July 2024
    License: MIT License
    """
    def __init__(self, n_estimators=10, 
                 estimator=DecisionTreeClassifier(),
                 population_size=50, num_generations=25, 
                 alpha=0.5, beta=0.5, random_state=None):
        super().__init__(n_estimators=n_estimators, 
                         estimator=estimator, 
                         random_state=random_state)
        self.population_size = population_size
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        if alpha + beta != 1:
            warnings.warn("alpha+beta should sum to 1. Normalizing...")
            total = alpha + beta
            self.alpha = alpha / total
            self.beta = beta / total
        self.best_fitness = None
        self.best_sol = []
        self.majority_data = None
        self.minority_data = None
        self.Lmaj = None
        self.Lmin = None

    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        params.update({
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "alpha": self.alpha,
            "beta": self.beta
        })
        if deep and hasattr(self.estimator, "get_params"):
            base_params = self.estimator.get_params(deep=deep)
            for key, value in base_params.items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **parameters):
        estimator_params = {}
        for key, value in parameters.items():
            if key.startswith("estimator__"):
                estimator_params[key.split('__', 1)[1]] = value
            else:
                setattr(self, key, value)
        if estimator_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**estimator_params)
        return self

    def __fitness_func(self, ga_instance, solution, solution_idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids = np.where(solution == 1)[0]
            if len(ids) == 0:
                return -np.inf
            X = np.vstack((self.majority_data[ids, :-1], 
                         self.minority_data[:, :-1]))
            y = np.concatenate((self.majority_data[ids, -1], 
                              self.minority_data[:, -1]))
            clf = clone(self.estimator)
            gmean_list = []
            unique, counts = np.unique(y, return_counts=True)
            n_classes = len(unique)
            if n_classes < 2:
                return -np.inf
            min_count = min(counts)
            if min_count < 5 and min_count > 1:
                cv = StratifiedKFold(n_splits=min_count)
            elif min_count == 1:
                clf.fit(X, y)
                y_pred = clf.predict(X)
                gmean = geometric_mean_score(y, y_pred)
                gmean_list.append(gmean)
            else:
                cv = StratifiedKFold(n_splits=5)
            if min_count > 1:
                for train_idx, test_idx in cv.split(X, y):
                    clf_cv = clone(self.estimator)
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    clf_cv.fit(X_train, y_train)
                    y_pred = clf_cv.predict(X_test)
                    gmean = geometric_mean_score(y_test, y_pred)
                    gmean_list.append(gmean)
            avg_gmean = np.mean(gmean_list) if gmean_list else 0
            diversity_term = 0
            if self.beta > 0:
                diversity = []
                for idx in range(self.population_size):
                    if idx == solution_idx:
                        continue
                    other_sol = ga_instance.population[idx]
                    N00 = np.sum((solution == 0) & (other_sol == 0))
                    N11 = np.sum((solution == 1) & (other_sol == 1))
                    N01 = np.sum((solution == 0) & (other_sol == 1))
                    N10 = np.sum((solution == 1) & (other_sol == 0))
                    denom = N11 * N00 + N10 * N01
                    if denom == 0:
                        diversity.append(0)
                    else:
                        diversity.append((N11 * N00 - N10 * N01) / denom)
                diversity_term = np.mean(diversity) if diversity else 0
            balance_term = abs(1 - len(self.minority_data) / len(ids))
            return avg_gmean - self.alpha * balance_term - self.beta * diversity_term

    def __on_generation(self, ga_instance):
        current_solution, current_fitness, _ = ga_instance.best_solution()
        self.best_fitness = current_fitness
        self.best_sol = current_solution
        print(f"The best fitness value: {self.best_fitness}")

    def __make_evolutionary_bootstrap(self, d_majority, d_minority):
        dc = {}
        rng = check_random_state(self.random_state)
        num_keep = round(0.2 * self.population_size)
        self.majority_data = d_majority
        self.minority_data = d_minority
        for b in range(self.n_estimators):
            # Handle random state properly for pygad
            random_seed = None if self.random_state is None else rng.randint(0, 2**32 - 1)
            
            ga_instance = pygad.GA(
                num_generations=self.num_generations,
                num_parents_mating=int(0.65 * self.population_size),
                fitness_func=self.__fitness_func,
                sol_per_pop=self.population_size,
                num_genes=len(d_majority),
                gene_type=int,
                init_range_low=0,
                init_range_high=2,
                mutation_by_replacement=True,
                parent_selection_type="rws",
                mutation_percent_genes=5,
                crossover_probability=0.9,
                mutation_probability=0.1,
                on_generation=self.__on_generation,
                crossover_type="two_points",
                keep_parents=num_keep,
                random_seed=random_seed,
                suppress_warnings=True
            )
            print(f"Evolutionary selection for the base learner: {b + 1}")
            ga_instance.run()
            solution, _, _ = ga_instance.best_solution()
            idx = np.where(solution == 1)[0]
            if len(idx) == 0:
                idx = rng.choice(len(d_majority), size=1, replace=False)
            d_majority_sampled = d_majority[idx, :]
            b_samp = np.vstack([d_majority_sampled, d_minority])
            dc[f'boot_{b}'] = {'boot': b_samp}
        return dc

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("EUSBag requires exactly two classes")
        self.classes_ = unique_classes
        self.n_features_in_ = X.shape[1]
        
        # Count class occurrences safely
        class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
        if min(class_counts.values()) < 2:
            raise ValueError("Each class must have at least 2 samples")
            
        # Determine minority/majority classes
        if class_counts[unique_classes[0]] > class_counts[unique_classes[1]]:
            self.Lmaj = unique_classes[0]
            self.Lmin = unique_classes[1]
        else:
            self.Lmaj = unique_classes[1]
            self.Lmin = unique_classes[0]
            
        idx_min = np.where(y == self.Lmin)[0]
        d_minority = np.concatenate((X[idx_min, :], y[idx_min].reshape(-1, 1)), axis=1)
        idx_maj = np.where(y == self.Lmaj)[0]
        d_majority = np.concatenate((X[idx_maj, :], y[idx_maj].reshape(-1, 1)), axis=1)
        dcBoot = self.__make_evolutionary_bootstrap(d_majority, d_minority)
        self.models = []
        for b in dcBoot:
            model = clone(self.estimator)
            boot_data = dcBoot[b]['boot']
            model.fit(boot_data[:, :-1], boot_data[:, -1].reshape(-1, 1))
            self.models.append(model)
        return self