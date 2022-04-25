import numpy as np
from scipy.special import comb
from scipy.spatial.distance import cdist
import time

def pairing_matrix(labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
    """

    INPUT:

    * labels1 - ndarray (n, ), partition number 1 with k clusters
    * labels2 - ndarray (n, ), partition number 2 with l clusters

    OUTPUT:

    * matrix - ndarray (k, l), contingency matrix of two partitions
    """

    assert labels1.shape[0] == labels2.shape[0]

    # get information about partitions
    unique1, ids1 = np.unique(labels1, return_inverse=True)
    unique2, ids2 = np.unique(labels2, return_inverse=True)

    # get number of clusters in each partitions
    size1 = unique1.shape[0]
    size2 = unique2.shape[0]

    # create contingency matrix
    matrix = np.zeros((size1, size2))

    # full it
    for i, j in zip(ids1, ids2):
        matrix[i, j] += 1

    return matrix


def centering(data: np.ndarray, g: np.ndarray = None, normalize: bool = False) -> np.ndarray:
    """
    INPUT:

    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * g - (optional) ndarray (1, m), center of mass for the given data matrix
    * normalize - (optional) bool, if True, then each columns of data matrix is MinMax Scaled between -1 and 1

    OUTPUT:

    * data - ndarray (n, m), centered (and normalized/scaled) data matrix, where n is the number of observations and m is the number of dimensions

    """
    # check if center of mass is given
    if g is None:

        # normalize if True
        if normalize:
            # normilized x = (x - min(x))/(max(x) - min(x))
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            data = np.nan_to_num(data, nan = 0.0)
        # center the data
        data = data - data.mean(axis=0)

    else:
        if normalize:
            g = (g - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            g = np.nan_to_num(g, nan = 0.0)
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            data = np.nan_to_num(data, nan = 0.0)
        data = data - g
    
    return data


def wss(Y: np.ndarray, X: np.ndarray) -> float :
    """
    INPUT:

    * Y - ndarray (n, ), where n is the size of dataset; the array of labels of partition
    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions

    OUTPUT:

    * total_wss - float, sum of squares of distances to the closest center (intertia)
    """
    unique = np.unique(Y)
    total_wss = 0.0
    for i in unique:
        centroid = np.mean(X[np.where(Y == i)], axis=0)
        distances = cdist(X[np.where(Y == i)], np.array([centroid]))
        distances=distances**2
        total_wss += distances.sum()
    return total_wss


def wcd(Y: np.ndarray, X: np.ndarray) -> float:
    """
    INPUT:

    * Y - ndarray (n, ), where n is the size of dataset; the array of labels of partition
    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions

    OUTPUT:

    * total_wcd - float, sum of distances to the closest centers
    """
    unique = np.unique(Y)
    total_wcd = 0.0
    for i in unique:
        centroid = np.mean(X[np.where(Y == i)], axis=0)
        distances = cdist(X[np.where(Y == i)], np.array([centroid]))
        total_wcd += distances.sum()
    return total_wcd


def calculate_inertia(X: np.ndarray, labels: np.ndarray, metric: str = 'wss'):
    """
    INPUT:

    * X -  ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * labels - ndarray(k, j, n), 3d matrix with partitions of length n for each of j executions for each value of number of clusters (k)
    * metric - string, 'wss' to calculate WSS for each partition, 'wcd' to calculate WCD for each partition

    OUTPUT:
    * metric_matrix - (k, j), matrix with values of the chosen metric for each partition
    """
    start = time.process_time()
    metric_matrix = np.zeros((labels.shape[0], labels.shape[1]))
    for k in range(labels.shape[0]):
        for ex in range(labels.shape[1]):
            if metric == 'wss':
                metric_matrix[k, ex] = wss(labels[k, ex], X)
            else:
                metric_matrix[k, ex] = wcd(labels[k, ex], X)
    
    stop = time.process_time()
    execution_time = stop - start
    
    return metric_matrix, execution_time

def get_1d_metric(matrix: np.ndarray, labels: np.ndarray, method: str = 'min'):
    """
    INPUT:

    * matrix -  ndarray (k, j), matrix with values of the chosen metric for each partition
    * labels - ndarray(k, j, n), 3d matrix with partitions of length n for each of j executions for each value of number
     of clusters (k)
    * method - string, 'min' to calculate and take partition with least value of metric for each value of number of cluster, 'mean' to calculate mean value of metric for each value of number of cluster and take the partition with least value of the metric

    OUTPUT:

    * metric_array - array with calculated value of metric
    * min_labels, ndarray (k, n), matrix with partition for each value of number of clusters
    """
    start = time.process_time()
    metric_array = np.zeros(labels.shape[0])
    min_labels = np.zeros((labels.shape[0], labels.shape[2]))
    for k in range(labels.shape[0]):
        if method == 'min':
            min_n = matrix[k].argmin()
            metric_array[k] = matrix[k, min_n]
            min_labels[k] = labels[k, min_n]
        else:
            min_n = matrix[k].argmin()
            metric_array[k] = matrix[k].mean()
            min_labels[k] = labels[k, min_n]
    stop = time.process_time()
    execution_time = stop - start
    return metric_array, min_labels, execution_time
