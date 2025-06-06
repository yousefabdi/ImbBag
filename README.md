# 📊 ImbBag

Imbalanced Bagging Ensemble Algorithms

## 📜 Description

ImbBag is a specialized package that integrates a variety of bagging ensemble methods specifically designed for imbalanced data classification. This package provides a scikit-learn-based framework that simplifies the usage of these methods, making it easier for researchers and practitioners to apply them in their work, whether dealing with binary or multi-class classification problems.

## 🛠 Installation

```bash
pip install imbbag
```

## ⚠️ Requirements

The following Python packages are required.

* scikit-learn
* imblearn 
* PyGAD 
* ARFS 
* mlxtend
* scikit-learn-intelex
* ARFS
* patch_sklearn

Also, use Python 3.11

## 🔍 Available Bagging Ensemble Algorithms in the ImbBag Package

* UnderBagging (UnderBag)
  * Multi-class
* Exactly Balanced Bagging (EBBag)
  * Binary-class
* OverBagging (OverBag)
  * Multi-class
* SMOTE Bagging (SMOTEBag)
  * Multi-class
* Roughly Balanced Bagging  (RBBag)
  * Binary-class
* Multi-class Roughly Balanced Bagging (MRBBag)
  * Multi-class
* Bagging Ensemble Variation (BEV)
  * Binary-class
* Lazy Bagging (LazyBag)
  * Multi-class
* Multi Random Balance Bagging (MultiRandBalBag)
  * Multi-class
* Neighborhood Balanced Bagging (NBBag)
  * Binary-class
* Probability Threshold Bagging (PTBag)
  * Multi-class
* Adaptive Synthetic Bagging (ADASYNBag)
  * Binary-class
* RSYN Bagging (RSYNBag)
  * Binary-class
* Resampling Ensemble Algorithm (REABag)
  * Multi-class
* Under-bagging K-NN (UnderBagKNN)
  * Multi-class
* Boundary Bagging (BBag)
  * Multi-class
* Bagging of Extrapolation-SMOTE SVM (BEBS)
  * Binary-class
* Evolutionary Under-sampling based Bagging (EUSBag)
  * Binary-class
* Random Balanced Sampling with Bagging (RBSBag)
  * Multi-class
* Cost-sensitive Bagging (CostBag)
  * Multi-class

## ⚡️ Get Started

Here is an example:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn. model_selection import train_test_split
from ImbBag import BBag

dataframe = read_csv('dataset.csv')
data = dataframe.values    
X = data[:,:-1]
Y = data[:,:-1]

# split the dataset into training and test sets
X_train ,X_test ,y_train ,y_test = train_test_split (X, y, test_size =0.2)

# instantiate the imbalance bagging classifier, training, prediction 
cls = BBag(estimator = DecisionTreeClassifier(), n_estimator = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
```

## 📅 Version History

- **v1.1.0** - Initial release - October 17, 2024
- **v1.3.0** - Updated relese  - June 06, 2025

## 🧑‍💻 Credits

- ** Yousef Abdi 
- *University of Tabriz*


## ⚖️ License

This project licensed under the MIT License.


## 💬 Support

Report issues, ask questions, and provide suggestions using:

* [GitHub Issues](https://github.com/yousefabdi/ImbBag/issues)
* [GitHub Discussions](https://github.com/yousefabdi/ImbBag/discussions)
* Email: y.abdi [at] tabrizu [dot] ac [dot] ir

The project can be accessed at https://github.com/yousefabdi/imbbag
