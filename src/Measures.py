#---------------------------------------------------------------
__author__ = "Yousef Abdi"
__email__ = "yousef.abdi@gmail.com"
__version__ = "0.0.1"
#---------------------------------------------------------------    
import numpy as np
class measurements:
    def __init__(self, ncols, nrows):
        self.gmeans = np.empty((ncols, nrows))
        self.kappa = np.empty((ncols, nrows))
        self.mauc = np.empty((ncols, nrows))
        self.f1 = np.empty((ncols, nrows))
        self. precision = np.empty((ncols, nrows))
        self.recall = np.empty((ncols, nrows))
        self.mmcc = np.empty((ncols, nrows))