import numpy as np
from scipy.spatial.distance import cdist


def one_hot(Y: np.ndarray, n_cat: int = None):
    return (Y.reshape(-1, 1) == np.arange(0, np.unique(Y).shape[0]).reshape(-1, np.unique(Y).shape[0])).astype(int) \
        if n_cat is None else (Y.reshape(-1, 1) == np.arange(0, n_cat).reshape(-1, n_cat)).astype(int)


# ex-wss
def sse(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None,
        power: float = 2, metric: str = 'euclidean', **kwargs) -> float:
    """
    Calculate the sum of squared errors between the data points and the cluster centers.

    The sum of squared errors (SSE) is a commonly used metric to evaluate the performance of clustering algorithms.
    It calculates the sum of the squared distances between each data point and its closest cluster center.

    Parameters:
        X (np.ndarray)
            The data points, with shape (N, D), where N is the number of samples and D is the number of features.
        Y (np.ndarray, optional)
            The binary indicator matrix, with shape (N, K), where K is the number of clusters.
        centers (np.ndarray, optional): The cluster centers, with shape (K, D).
            If not provided, the cluster centers are estimated as the mean of the data points in each cluster.
        power: float
            Value to power the sum of distances
        metric: str
            Metric to be used to compute distance
        **kwargs : Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.
    Returns:
        float: The sum of squared errors.
    """

    if centers is None:
        assert Y is not None
        assert Y.shape[0] == X.shape[0], f'Different shape: Y is {Y.shape}, X is {X.shape}'
        if len(Y.shape) == 1:
            Y = one_hot(Y)
        centers = (np.matmul(X.T, Y) / Y.sum(axis=0)).T
    assert centers.shape[1] == X.shape[1]

    return np.power(cdist(X, centers, metric=metric, **kwargs), power).min(axis=1).sum()


def pairing_matrix(labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
    """
    Calculate pairing matrix between two partitions.

    Parameters
    ----------
    labels1 : np.ndarray
        An array of integers representing the first partition.
    labels2 : np.ndarray
        An array of integers representing the second partition.

    Returns
    -------
    np.ndarray
        A pairing matrix between the two partitions.

    Examples
    --------
    >> labels1 = np.array([0, 0, 1, 2, 2, 2])
    >> labels2 = np.array([1, 1, 2, 0, 0, 0])
    >> pairing_matrix(labels1, labels2)
    array([[0., 2., 1.],
           [0., 0., 1.],
           [3., 0., 0.]])
    """
    assert labels1.shape[0] == labels2.shape[0], 'Input arrays should have the same length.'

    unique1, ids1 = np.unique(labels1, return_inverse=True)
    unique2, ids2 = np.unique(labels2, return_inverse=True)

    size1 = unique1.shape[0]
    size2 = unique2.shape[0]

    matrix = np.zeros((size1, size2), dtype=int)
    np.add.at(matrix, (ids1, ids2), 1)

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
