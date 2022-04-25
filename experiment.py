import numpy as np
import time
import matplotlib.pyplot as plt
from .kmeans import KMeans
from .miscellaneous import centering, calculate_inertia
'''
from kmeans import KMeans
from miscellaneous import centering, calculate_inertia
'''
class KMeansMultiple(object):
    """

    """

    def __init__(self, X: np.ndarray, k_range: np.ndarray = np.arange(1, 26), execution_number: int = 10):
        self.data = X
        self.gravity_center = None
        self.max_iter = 50
        self.init_method = 'maxmin_min'
        self.init_centers = None
        self.k_range = k_range
        self.execution_number = execution_number
        self.L = np.zeros((k_range.shape[0], execution_number, X.shape[0]))
        self.normalize = True
        self.execution_time = None

    def change_default_params(self, max_iter: int = 50, init_method: str = 'maxmin_min',
                              init_centers: np.ndarray = None,
                              gravity_center: np.ndarray = None, normalize: bool = True):
        self.max_iter = max_iter
        self.init_method = init_method
        self.init_centers = init_centers
        self.normalize = normalize
        self.gravity_center = gravity_center

        return self

    def execute(self):
        start = time.process_time()
        x = centering(data = self.data, g=self.gravity_center, normalize=self.normalize)
        for k in range(self.k_range.shape[0]):
            for iteration in range(self.execution_number):
                self.L[k, iteration] = KMeans(X=x, k=self.k_range[k], max_iter=self.max_iter) \
                    .fit(self.init_centers, self.init_method).labels
        stop = time.process_time()
        self.execution_time = stop - start

        return self

