# imports
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import time


def centers_initialization(X: np.ndarray, k: int, method: str):
    """
    INPUT:

    *  X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    *  k - int,  the number of clusters
    *  method -  str, 'maxmin_min' for MaxMin by Minimal Distance initialization, 'maxmin_sum' for MaxMin by Minimal Sum

    OUTPUT:

    *  init_centers - ndarray (k, m), data matrix, where rows are the initialized centers
    """

    init_centers = np.zeros((k, X.shape[1]))
    # 'maxmin_min
    if method == 'maxmin_min':
        for i in range(k):
            if i == 0:
                cen_i = np.random.randint(0, X.shape[0])
            else:
                cen_i = np.min(cdist(X, init_centers[:i]), axis=1).argmax()
            init_centers[i] = X[cen_i]
    else:
        # 'maxmin_sum'
        temp = []
        for i in range(k):
            if i == 0:
                cen_i = np.random.randint(0, X.shape[0])
                init_centers[i] = X[cen_i]
                temp.append(cen_i)
            else:
                cen_i = np.sum(cdist(init_centers[:i], X), axis=0).argmax()
                if cen_i in temp:
                    cen_i = np.sum(np.absolute(
                        cdist(init_centers[:i], X) - np.reshape(np.mean(cdist(init_centers[:i], X), axis=0),
                                                                (1, X.shape[0]))), axis=0).argmin()

                init_centers[i] = X[cen_i]
                temp.append(cen_i)

    return init_centers


def kmeans(data: np.ndarray, k: int, init_centers: np.ndarray = None, init_method: str = 'maxmin_min',
           max_iter: int = 50, save_history: bool = False):
    """
    INPUT:

    * data  - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * k - int,  the number of clusters * init_centers - (optional) ndarray (k, m), data matrix, where rows are the initial centers
    * init_method -  (optional) str, 'maxmin_min' by default - for MaxMin by Minimal Distance initialization, 'maxmin_sum' for MaxMin by Minimal Sum
    * max_iter - (optional) int, the number of maximal iterations of the algorithm
    * save_history - (optional) boolean, if True, output the list history with labels of  each iteration

    OUTPUT:
    * labels - array (1, n), list of labels after the convegance, where n is number of observations
    * centers - ndarray (k, m), matrix, where rows are the centers of the final clusters
    * history - (optional) list of length L, that contains L arrays of labels of each iteration

    """

    start = time.process_time()
    # define number observations
    n = data.shape[0]
    # define number of dimensions
    m = data.shape[1]

    assert k >= 1

    if k < 2:
        return np.zeros(data.shape[0]), list(np.mean(data, axis=0)), 0

    # Check centers
    if init_centers is not None:
        assert k == len(init_centers)
        assert np.array(init_centers).shape[1] == m
    else:
        init_centers = centers_initialization(data, k, init_method)

    if save_history:
        history = []

    labels = np.zeros(n)
    old_centers = np.array(init_centers)
    iteration = 0

    # convergence process
    while True:

        # init parameters
        new_centers = []
        iteration += 1

        # distances
        distances = distance_matrix(data, old_centers)

        assert distances.shape[0] == n
        assert distances.shape[1] == k

        labels = distances.argmin(axis=1)
        assert len(labels) == n

        if save_history == True:
            history.append(labels)

        for j in range(k):
            new_centers.append(np.mean(data[np.where(labels == j)], axis=0))

        if np.all(np.array(new_centers) == np.array(old_centers)) != True:
            old_centers = new_centers
        else:
            break
        if iteration > max_iter:
            break

    centers = new_centers

    stop = time.process_time()
    execution_time = stop - start

    # output
    if save_history == True:
        return labels.astype(int), centers, execution_time, history
    else:
        return labels.astype(int), centers, execution_time


def execute_kmeans(X: np.ndarray, k_values: np.ndarray = np.arange(1, 26), execution_number: int = 10,
                   init_method='maxmin_min'):
    """
    INPUT:

    * X - ndarray(N, M), where N is number of observations and M is number of dimensions, input dataset
    * k_values - ndarray(k), with values of number of clusters to calculate
    * execution_number - int, number of executions for each value of number of clusters
    * init_method - string, the method of initialization of centers for k-means

    OUTPUT:
    * labels - ndarray(k, j, N), 3d matrix with resulting partitions
    """
    start = time.process_time()

    labels = np.zeros((k_values.shape[0], execution_number, X.shape[0]))

    for k in range(k_values.shape[0]):
        for iter_n in range(execution_number):
            labels[k, iter_n], _, _ = kmeans(X, k=k_values[k], init_method=init_method)

    stop = time.process_time()
    execution_time = stop - start

    return labels, execution_time


class KMeans(object):
    """
    """

    def __init__(self, X: np.ndarray, k: int, max_iter: int = 50, save_history: bool = False):
        self.data = X
        self.k = k
        self.max_iter = max_iter
        self.save_history = save_history
        if save_history:
            self.history = []
        self.iteration = None
        self.init_centers = None
        self.centers = None
        self.labels = None
        self.execution_time = None

    def fit(self, init_centers: np.ndarray = None, init_method: str = 'maxmin_min'):

        start = time.process_time()
        # define number observations
        n = self.data.shape[0]
        # define number of dimensions
        m = self.data.shape[1]

        assert self.k >= 1

        if self.k < 2:
            self.labels = np.zeros(self.data.shape[0])
            self.centers = list(np.mean(self.data, axis=0))
            self.execution_time = 0
            return self

        # Check centers
        if init_centers is not None:
            assert self.k == len(init_centers)
            assert np.array(init_centers).shape[1] == m
            self.init_centers = init_centers
        else:
            self.init_centers = centers_initialization(self.data, self.k, init_method)

        labels = np.zeros(n)
        old_centers = np.array(self.init_centers)
        self.iteration = 0

        # convergence process
        while True:

            # init parameters
            new_centers = []
            self.iteration += 1

            # distances
            distances = distance_matrix(self.data, old_centers)

            assert distances.shape[0] == n
            assert distances.shape[1] == self.k

            labels = distances.argmin(axis=1)
            assert len(labels) == n

            if self.save_history:
                self.history.append(labels)

            for j in range(self.k):
                new_centers.append(np.mean(self.data[np.where(labels == j)], axis=0))

            if not np.all(np.array(new_centers) == np.array(old_centers)):
                old_centers = new_centers
            else:
                break
            if self.iteration > self.max_iter:
                break

        self.centers = new_centers
        self.labels = labels.astype(int)
        stop = time.process_time()
        self.execution_time = stop - start

        return self
