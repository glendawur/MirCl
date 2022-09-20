import numpy as np
from scipy.spatial.distance import cdist


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
            data = np.nan_to_num(data, nan=0.0)
        # center the data
        data = data - data.mean(axis=0)

    else:
        if normalize:
            g = (g - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            g = np.nan_to_num(g, nan=0.0)
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
            data = np.nan_to_num(data, nan=0.0)
        data = data - g

    return data

# ex-wss
def sse(Y: np.ndarray, X: np.ndarray) -> float:
    """
    INPUT:
    * Y - ndarray (n, ), where n is the size of dataset; the array of labels of partition
    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    OUTPUT:
    * total_wss - float, sum of squares of distances to the closest center (intertia)
    """
    unique = np.unique(Y)
    n = Y.shape[0]
    m = X.shape[1]
    total_sse = 0.0
    for i in unique:
        centroid = np.mean(X[np.where(Y == i)], axis=0)
        distances = cdist(X[np.where(Y == i)].reshape((-1, m)), np.array([centroid]).reshape((-1, m)))
        distances=distances**2
        total_sse += distances.sum()
    return total_sse


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