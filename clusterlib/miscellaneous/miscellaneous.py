import numpy as np
from scipy.spatial.distance import cdist


def one_hot(Y: np.ndarray, n_cat: int = None):
    return (Y.reshape(-1, 1) == np.arange(0, np.unique(Y).shape[0]).reshape(-1, np.unique(Y).shape[0])).astype(int) \
        if n_cat is None else (Y.reshape(-1, 1) == np.arange(0, n_cat).reshape(-1, n_cat)).astype(int)


# ex-wss
def sse(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None) -> float:
    """
    Calculate the sum of squared errors between the data points and the cluster centers.

    The sum of squared errors (SSE) is a commonly used metric to evaluate the performance of clustering algorithms.
    It calculates the sum of the squared distances between each data point and its closest cluster center.

    Parameters:
    X (np.ndarray): The data points, with shape (N, D), where N is the number of samples and D is the number of features.
    Y (np.ndarray, optional): The binary indicator matrix, with shape (N, K), where K is the number of clusters.
    centers (np.ndarray, optional): The cluster centers, with shape (K, D).
    If not provided, the cluster centers are estimated as the mean of the data points in each cluster.

    Returns:
    float: The sum of squared errors.
    """
    if centers is None:
        assert Y is not None
        centers = (np.matmul(X.T, Y) / labels.sum(axis=0)).T

    return np.power(cdist(X, centers), 2).min(axis=1).sum()


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
    Subtract mean (centering) and normalize the data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
    g : numpy.ndarray, optional
        Mean value to be subtracted. If None, mean of `data` will be used.
    normalize : bool, optional
        If True, normalize the data.

    Returns
    -------
    numpy.ndarray
        Centered and normalized data.

    """
    if g is None:
        g = np.mean(data, axis=0)

    data = data - g
    if normalize:
        data = (data - data.mean(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data
