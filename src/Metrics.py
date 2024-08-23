import numpy as np

# This class implements HVDM that uses standard deviation to normalize distance
class HVDMstd(object):
    def __init__(self, data):
        self.X = data

    def __hvdm_numeric(a, b, std_dev):
        return np.abs(a - b) / std_dev

    def __hvdm_categorical(self, a, b, n, nc):
        if a == b:
            vdm = 0
        else:
            vdm = np.abs(nc[a]/n[a] - nc[b]/n[b])
        return vdm

    def hvdm(self, a, b):
        n = len(self.X)
        d = len(a)
        distance = 0
        for i in range(d):
            if isinstance(a[i], float):
                std_dev = np.std([row[i] for row in self.X])
                distance += self.__hvdm_numeric(a[i], b[i], std_dev)
            else:
                nc = {value: sum(1 for row in self.X if row[i] == value) for value in \
                      set(row[i] for row in self.X)}
                distance += self.__hvdm_categorical(a[i], b[i], n, nc)
        return distance
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
# This class implements plain HVDM metric
class HVDM(object):
    def __hvdm_categorical(x_i, x_j):
        n = len(set(x_i) | set(x_j))
        delta_ikj = np.sum(x_i != x_j)
        return delta_ikj / n
    
    def __hvdm_numerical(self, x_i, x_j, feature_range):
        if feature_range != 0:
            return np.abs(x_i - x_j) / feature_range
        else:
            return np.abs(x_i - x_j)


    def calculate_hvdm(self, instance_i, instance_j):
        num_features = len(instance_i)
        feature_range = np.ptp(instance_i)  # Range of numerical feature values (max - min)

        #Feature weights
        weights = np.ones(num_features)  # Equal weights for all features

        hvdm_sum = 0
        for f in range(num_features):
            if isinstance(instance_i[f], (int, float)):
                hvdm_sum += weights[f] * self.__hvdm_numerical(instance_i[f], instance_j[f], feature_range)
            else:
                hvdm_sum += weights[f] * self.__hvdm_categorical(instance_i[f], instance_j[f])

        return hvdm_sum
