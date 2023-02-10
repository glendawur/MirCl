import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# input: data X and labels Y

def wss(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
    """
        Calculates the Within Sum of Squared Error for the given samples X and their partition in Y or by using the passed centroids.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            The samples to be used in the calculation.
        Y : np.ndarray, shape (n_samples,), optional (default=None)
            The partition of the samples into classes. If not passed, `centers` must be passed.
        centers : np.ndarray, shape (n_clusters, n_features), optional (default=None)
            The centers of the clusters. If not passed, `Y` must be passed.
        method : str, optional (default='conventional')
            The method used for computing the WSS. The possible values are:
            - 'conventional': WSS is calculated using the conventional method.
            - 'euclidean': WSS is calculated as sum of distances

        Returns
        -------
        float
            The Within Sum of Squared Error.

        Raises
        ------
        AssertionError
            If both Y and centers are not passed.
            If `centers` is passed, but has a different number of features than X.
            If `Y` is passed, but has a different number of samples than X.

        Examples
        --------
        >> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >> Y = np.array([0, 0, 0, 1, 1, 1])
        >> wss(X, Y)
        16.0
        >> centers = np.array([[1, 2], [4, 2]])
        >> wss(X, centers=centers, 'euclidean')
        8.0
        """

    if centers is not None:
        assert (centers.shape[1] == X.shape[1])
        distances = cdist(X, centers).min(axis=1)
        if method == 'conventional':
            distances = distances ** 2
        return distances.sum()
    elif Y is not None:
        assert (Y.shape[0] == X.shape[0])
        unique = np.unique(Y)
        total_wss = 0.0
        for i in unique:
            centroid = np.mean(X[np.where(Y == i)], axis=0)
            distances = cdist(X[np.where(Y == i)], np.array([centroid]))
            if method == 'conventional':
                distances = distances ** 2
            total_wss += distances.sum()
        return total_wss
    else:
        print("Error, no partition passed")


# aux rewritten
def wss_axis(Y: np.ndarray, X: np.ndarray, method: str = 'conventional'): return wss(X=X, Y=Y, method=method)


def wss_matrix(X: np.ndarray, L: np.ndarray = None, method: str = 'conventional'):
    """
    Calculates the Within Sum of Squares (WSS) matrix for each partition in matrix L.

    Parameters
    ----------
        X : np.ndarray, shape (n_samples, n_features)
        Input data points.

        L : np.ndarray, shape (k, n_partitions, n_samples)
        The partition labels.

        method : str, optional (default = 'conventional')
        The method used for computing the WSS matrix. The possible values are:
        - 'conventional': WSS is calculated using the conventional method.
        - 'euclidean': WSS is calculated as sum of distances

    Returns
    ----------
        inertia : np.ndarray, shape (n_partitions, n_samples)
        The WSS matrix, where each row represents the WSS for each partition.

    """
    inertia = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        inertia[i] = np.apply_along_axis(wss_axis, 1, L[i], X, method).reshape(-1)
    return inertia


def bss(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
    """
    Calculate the between-group sum of squares (BSS) based on given data points, labels, or cluster centers.

    The between-group sum of squares measures the amount of variance between the different groups.

    Parameters
    ----------
        X : numpy.ndarray
        The data points as a 2-D array of shape (n_samples, n_features).

        Y : numpy.ndarray, optional
        The labels for the data points as a 1-D array of shape (n_samples,) (default is None).

        centers : numpy.ndarray, optional
        The cluster centers as a 2-D array of shape (n_clusters, n_features) (default is None).

        method : str, optional
        The method used to calculate the BSS and WSS, either 'conventional' or 'euclidean' (default is 'conventional').

    Returns
    ----------
        float
        The BSS value.

    Raises
    ----------
        AssertionError
        If Y is not None and Y.shape[0] is not equal to X.shape[0].
        If centers is not None and centers.shape[1] is not equal to X.shape[1
    """
    if centers is not None:
        assert (centers.shape[1] == X.shape[1])
        labels = cdist(X, centers).argmin(axis=1)
        total_bss = 0.0
        for i in range(centers.shape[0]):
            if method == 'conventional':
                total_bss += np.where(labels == i)[0].shape[0] * cdist(centers[i].reshape((1, -1)),
                                                                       X.mean(axis=0).reshape((1, -1))) ** 2
            else:
                total_bss += np.where(labels == i)[0].shape[0] * cdist(centers[i].reshape((1, -1)),
                                                                       X.mean(axis=0).reshape((1, -1)))
        return total_bss
    elif Y is not None:
        assert (Y.shape[0] == X.shape[0])
        total_bss = 0.0
        unique = np.unique(Y)
        for i in unique:
            centroid = np.mean(X[np.where(Y == i)], axis=0)
            if method == 'conventional':
                total_bss += np.where(Y == i)[0].shape[0] * cdist(centroid.reshape((1, -1)),
                                                                  X.mean(axis=0).reshape((1, -1))) ** 2
            else:
                total_bss += np.where(Y == i)[0].shape[0] * cdist(centroid.reshape((1, -1)),
                                                                  X.mean(axis=0).reshape((1, -1)))
        return total_bss
    else:
        print("Error, no partition passed")


def bss_axis(Y: np.ndarray, X: np.ndarray, method: str = 'conventional'): return bss(X=X, Y=Y, method=method)


def bss_matrix(X: np.ndarray, L: np.ndarray = None, method: str = 'conventional'):
    """
    Calculates the Between-group Sum of Squares (BSS) matrix for each partition in matrix L.

    Parameters
    ----------
        X : np.ndarray, shape (n_samples, n_features)
        Input data points.

        L : np.ndarray, shape (k, n_partitions, n_samples)
        The partition labels.

        method : str, optional (default = 'conventional')
        The method used for computing the WSS matrix. The possible values are:
        - 'conventional': BSS is calculated using the conventional method.
        - 'euclidean': BSS is calculated as sum of distances

    Returns
    ----------
        inertia : np.ndarray, shape (n_partitions, n_samples)
        The BSS matrix, where each row represents the BSS for each partition.
    """

    inertia = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        inertia[i] = np.apply_along_axis(bss_axis, 1, L[i], X, method).reshape(-1)
    return inertia


def calinski_harabasz(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
    """
       Calculate the Calinski-Harabasz index for a given clustering partition.

       The Calinski-Harabasz index is a measure of the quality of a clustering algorithm. It is defined as the ratio between the
       between-cluster sum of squares and the within-cluster sum of squares. The index increases as the ratio of between-cluster
       sum of squares to within-cluster sum of squares increases.

       Parameters
       ----------
           X : np.ndarray
           Input data, a 2D array of shape (n_samples, n_features).

           Y : np.ndarray, optional
           An array of shape (n_samples,) with the cluster labels for each sample. If not provided, centers must be provided.
           The default is None.

           centers : np.ndarray, optional
           An array of shape (n_clusters, n_features) with the cluster centroids. If not provided, Y must be provided. The
           default is None.

           method : str, optional
           The method to use for the calculation of the within-cluster sum of squares. Can be either 'conventional' or 'euclidean'.
           The default is 'conventional'.

       Returns
       ----------
           float
           The Calinski-Harabasz index for the given clustering partition.

       Raises
       ----------
           ValueError
           If both Y and centers are not provided or both are provided.

        References
        ----------
            Calinski, T. & Harabasz, J. A dendrite method for cluster analysis. Commun Stat-Simulat Comput 3, 1–27 (1974).
       """
    if Y is None:
        assert centers is not None
        cls_num = centers.shape[0]
    elif centers is None:
        assert (Y is not None) & (Y.shape[0] == X.shape[0])
        cls_num = np.unique(Y).shape[0]
    else:
        print("Error, no partition passed")
        return None
    return (bss(X, Y, centers, method) / (cls_num - 1)) / (wss(X, Y, centers, method) / (X.shape[0] - cls_num))


def count(X: np.ndarray): return np.unique(X).shape[0]


def calinski_harabasz_matrix(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, SSB: np.ndarray, aggregation=np.mean):
    """
        Matrix version of calinski_harabasz

        Parameters
        ----------
            X : np.ndarray, shape (n_samples, n_features)
            Input data points.

            L : np.ndarray, shape (k, n_partitions, n_samples)
            The partition labels.

            SSW : np.ndarray
            Sum of (squared) distances within each cluster.

            SSB : np.ndarray
            Sum of (squared) distances between each cluster.

            aggregation : function, optional
            The aggregation function to use when calculating the SSW/SSB ratio at each number of clusters, by default np.mean.
            If None, no aggregation applied

        Returns
        ----------
            np.ndarray, shape (k, n_partitions) or (k, )
            The Calinski-Harabasz matrix or array of aggregated values

        """
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return (SSB / (classes - 1)) / (SSW / (X.shape[0] - classes))
    else:
        aggregated = aggregation(SSB/SSW, axis = 1)
        return aggregated*((X.shape[0] - np.mean(classes, axis = 1))/(np.mean(classes, axis = 1) - 1))

def xu_index(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None):
    """
        Calculates the Xu index of a cluster partition of data.
        The Xu index ranges from 0 to infinity and a high value indicates a good quality clustering solution.

        The Xu index is a clustering validation index that measures the quality of a clustering solution.
        It was introduced by Xu in 1998 and is based on the idea of minimizing the sum of the variance of the distances between data points and their cluster centroids. The formula for the Xu index is:

        Xu_index = (B_ss / (k-1)) / (W_ss / (n-k)),
        where B_ss is the between-cluster sum of squares, W_ss is the within-cluster sum of squares,
         k is the number of clusters, and n is the number of data points.



        Parameters
        ----------
            X : np.ndarray, shape (n_samples, n_features)
                The data matrix.
            Y : np.ndarray, optional, shape (n_samples,)
                The class labels of the data.
            centers : np.ndarray, optional, shape (n_clusters, n_features)
                The cluster centers, one center per row.

        Returns
        -------
            xu : float
                The Xu index of the cluster partition.

        Notes
        -----
            Either the class labels or the cluster centers must be passed to the function.

        References
        ----------
            Xu, R. (1998). Extracting the cluster validity index from the validated fuzzy clustering models. Fuzzy Sets and Systems, 90(1), 119-124.
    """

    if Y is None:
        assert centers is not None
        cls_num = centers.shape[0]
    elif centers is None:
        assert (Y is not None) & (Y.shape[0] == X.shape[0])
        cls_num = np.unique(Y).shape[0]
    else:
        print("Error, no partition passed")
        return None
    return X.shape[1] * np.log(np.sqrt(wss(X, Y, centers) / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(cls_num)


def xu_index_matrix(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, aggregation=np.mean):
    """
            Matrix version of xu_index

            Parameters
            ----------
                X : np.ndarray, shape (n_samples, n_features)
                Input data points.

                L : np.ndarray, shape (k, n_partitions, n_samples)
                The partition labels.

                SSW : np.ndarray
                Sum of (squared) distances within each cluster.

                aggregation : function, optional
                The aggregation function to use when calculating the SSW at each number of clusters, by default np.mean.
                If None, no aggregation applied

            Returns
            ----------
                np.ndarray, shape (k, n_partitions) or (k, )
                The Xu index matrix or array of aggregated values

            """
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return X.shape[1] * np.log(np.sqrt(SSW / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(classes)
    else:
        return X.shape[1] * np.log(np.sqrt(aggregation(SSW, axis = 1) / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(np.mean(classes, axis = 1))


def wb_index(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None):
    """
    WB index (Wu and Bailey index)

    The WB index is a validation index for cluster analysis that balances between between-cluster variability (BSS)
     and within-cluster variability (WSS).

    Parameters:
    ----------
        X : numpy.ndarray
            Data matrix to be partitioned. It should be of shape `(n_samples, n_features)`

        Y : numpy.ndarray, optional, default = None
            Array of true labels. It should be of shape `(n_samples,)`

        centers : numpy.ndarray, optional, default = None
            Array of cluster centers. It should be of shape `(n_clusters, n_features)`

    Returns:
    ----------
        wb_index : float
            The WB index for the given partition of data

    References:
    ----------
        Wu, X. and Bailey, K. (1994). A linear index for validity of fuzzy clustering. IEEE Transactions on Fuzzy Systems, 2(1), p.74-79.
    """
    if Y is None:
        assert centers is not None
        cls_num = centers.shape[0]
    elif centers is None:
        assert (Y is not None) & (Y.shape[0] == X.shape[0])
        cls_num = np.unique(Y).shape[0]
    else:
        print("Error, no partition passed")
        return None
    return cls_num * wss(X, Y, centers) / bss(X, Y, centers)


def wb_index_matrix(L: np.ndarray, SSW: np.ndarray, SSB: np.ndarray, aggregation=np.mean):
    """
            Matrix version of wb_index

            Parameters
            ----------
                L : np.ndarray, shape (k, n_partitions, n_samples)
                The partition labels.

                SSW : np.ndarray
                Sum of (squared) distances within each cluster.

                SSB : np.ndarray
                Sum of (squared) distances between each cluster.

                aggregation : function, optional
                The aggregation function to use when calculating the SSW/SSB ratio at each number of clusters, by default np.mean.
                If None, no aggregation applied

            Returns
            ----------
                np.ndarray, shape (k, n_partitions) or (k, )
                The WB index matrix or array of aggregated values

            """
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return classes * SSW / SSB
    else:
        aggregated = aggregation(SSW/SSB, axis = 1)
        return np.mean(classes, axis = 1)*aggregated
        
        
def silhouette(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None,
               result: str = 'mean'):
    """
    Calculate the Silhouette index for a given clustering.

    The Silhouette index is a measure of how well each sample has been assigned to its own cluster, compared to other clusters.
    The score ranges between -1 and 1, where a high value indicates that the sample is well-matched to its own cluster and poorly matched to neighboring clusters.

    Parameters:
    ----------
        X (np.ndarray): The data array with shape (n_samples, n_features).
        Y (np.ndarray, optional): The target array, with shape (n_samples,)
        centers (np.ndarray, optional): The array of cluster centroids, with shape (n_clusters, n_features).
        result (str, optional): Determines whether to return the mean silhouette score for all samples or the individual silhouette score for each sample. Default is 'mean'.

    Returns:
    ----------
        float or np.ndarray: The Silhouette index. If result is 'mean', a float is returned. If result is 'each', a numpy array with shape (n_samples,) is returned.

    References:
    ----------
        Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.
    """
    if Y is None:
        assert centers is not None
        labels = cdist(X, centers).argmin(axis=1)
    elif centers is None:
        assert (Y is not None) & (Y.shape[0] == X.shape[0])
        labels = Y.copy()
    else:
        print("Error, no partition passed")
        return None

    # intermediate step
    bool_matrix = np.tile(labels, (labels.shape[0], 1)) == np.tile(labels, (labels.shape[0], 1)).T
    np.fill_diagonal(bool_matrix, False)
    distances = cdist(X, X)
    u, c = np.unique(labels, return_counts=True)
    # array of a(i)
    a = np.sum(distances * bool_matrix, axis=1) / np.sum(bool_matrix, axis=1)
    # array of b(i)
    b = np.zeros(labels.shape[0])
    for obs in range(labels.shape[0]):
        distance = np.zeros(u.shape[0])
        for label in range(u.shape[0]):
            distance[label] = np.sum(distances[obs, np.where(labels == u[label])], axis=1) / c[label]
        distance[int(labels[obs])] = np.inf
        b[obs] = np.min(distance)

    return np.mean(np.nan_to_num((b - a) / np.max([a, b], axis=0)))


def silhouette_axis(Y: np.ndarray, X: np.ndarray, method: str = 'mean'):
    return silhouette(X=X, Y=Y, centers=None, result=method)


def silhouette_matrix(X: np.ndarray, L: np.ndarray, method: str = 'mean', aggregation=None):
    """
    Matrix version of silhouette

    Parameters
    ----------
        X : np.ndarray, shape (n_samples, n_features)
        Input data points.

        L : np.ndarray, shape (k, n_partitions, n_samples)
        The partition labels.

        aggregation : function, optional
        The aggregation function to use when calculating the Silhouette wiidth ratio at each number of clusters, by default None
        If None, no aggregation applied

    Returns
    ----------
        np.ndarray, shape (k, n_partitions) or (k, )
        The Calinski-Harabasz matrix or array of aggregated values

    """
    silhouettes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        silhouettes[i] = np.apply_along_axis(silhouette_axis, 1, L[i], X, method).reshape(-1)
    if aggregation is None:
        return silhouettes
    else:
        return aggregation(silhouettes, axis = 1)

def silhouette_wss(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, aggregation=np.argmin):
    """
    Calculates the silhouette score for each partition in the list of partitions shortcutting the large computations by
    the choice of single representative partition for each value of number of clusters based on SSW.

    Parameters
    ----------
        X : np.ndarray
            The data matrix, with shape `(n_samples, n_features)`.
        L : np.ndarray
            The list of partitions, with shape `(n_partitions, n_samples)`.
        SSW : np.ndarray
            The within-cluster sum of squares for each partition, with shape `(n_partitions,)`.
        aggregation : function, optional, default: `np.argmin`
            The method for to choose the single representative partition for the given number of clusters from L based on SSW.
            One of ['min', 'max']. The function should take two arguments: `(SSW, axis=1)`.
            The aggregation function for reducing the list of SSW to a single number for each partition.

    Returns
    -------
        silhouettes : np.ndarray
            The silhouette score for each partition, with shape `(n_partitions,)`.
    """
    assert aggregation is not None
    chosen_partitions = L[(np.arange(0, L.shape[0]), aggregation(SSW, axis=1))]
    silhouettes = np.zeros((L.shape[0],))
    for i in range(L.shape[0]):
        silhouettes[i] = silhouette(X, chosen_partitions[i])
    return silhouettes


# NOTE: add parameter method: 'adjusting' (not implemented) and 'fixed' () version
def elbow(SSW: np.ndarray, levels: (int, int) = (1, 1), aggregation=np.mean):
    """
    Calculate the elbow point for the given SSW array.

    The elbow point is defined as the point of inflection on the curve of SSW vs the number of clusters.
    The curve is smoothed using the levels parameter, and the position of the elbow point is then
    estimated based on the smoothed curve.

    Parameters
    ----------
        SSW : np.ndarray
            Sum of squared distances within each cluster.
        levels : tuple, optional
            The number of levels to use for the smoothing spline, by default (1, 1).
        aggregation : function, optional
            The aggregation function to use when calculating the SSW at each level, by default np.mean.

    Returns
    -------
        int
            The estimated number of clusters corresponding to the elbow point.

    Notes
    -------
        More variations of the method is coming soon
    """
    # if method == 'fixed':
    if aggregation is not None:
        aggregated = aggregation(SSW, axis=1)
        indices = np.full((SSW.shape[0],), -np.inf)
        frac = (aggregated[:-(levels[0] + levels[1])] - aggregated[levels[0]:-levels[1]]) / (
                aggregated[levels[0]:-levels[1]] - aggregated[(levels[0] + levels[1]):])
        indices[levels[0]:-levels[1]] = frac
    else:
        indices = np.full(SSW.shape, -np.inf)
        frac = (SSW[:-(levels[0] + levels[1])] - SSW[levels[0]:-levels[1]]) / (
                SSW[levels[0]:-levels[1]] - SSW[(levels[0] + levels[1]):])
        indices[levels[0]:-levels[1]] = frac
    return indices

def hartigan(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, aggregation=np.mean):
    """
    Hartigan index to evaluate the quality of clustering solutions

    The Hartigan index (Rule of Thumb)) is a clustering evaluation method that assesses the quality of
    the clustering solution by comparing the sum of squared distances within the clusters
    and the sum of squared distances between the clusters.

    Parameters
    ----------
        X : np.ndarray
            The data matrix with `n` observations and `p` features. Shape = (n, p).
        L : np.ndarray
            The cluster labels for each observation in `X`. Shape = (n,).
        SSW : np.ndarray
            The sum of squared (euclidean) distances within each cluster, for the solution represented by `L`.
        aggregation : callable, optional
            A callable that aggregates the results from multiple cluster solutions.
            The default is `np.mean`.

    Returns
    -------
        float
            The Hartigan index value, which ranges from 0 to 1, with higher values indicating
            a better clustering solution.

    References
    ----------
        Hartigan, J. A. (1975). Clustering algorithms. Wiley.
    """

    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is not None:
        aggregated = aggregation(SSW, axis=1)
        num_classes = np.mean(classes, axis=1)
        indices = np.full((SSW.shape[0],), np.inf)
        indices[:-1] = (aggregated[:-1] / aggregated[1:] - 1) * (X.shape[0] - num_classes[:-1] - 1)
    else:
        indices = np.full(SSW.shape, np.inf)
        indices[:-1] = (SSW[:-1] / SSW[1:] - 1) * (X.shape[0] - classes[:-1] - 1)
    return indices


def find_optimal(indices: np.ndarray, method: str, k_range: np.ndarray, to_plot: bool = False):
    """
    Find the optimal number of clusters based on given indices array and the method.

    Parameters
    ----------
        indices : numpy.ndarray
            Array with indices values.
        method : str
            Method to be used to compute the index. Supported indices:
            -'silhouette'
            -'elbow'-
            -'calinski_harabasz'
            -'hartigan'
            -'wb_index'
            -'xu_index
        k_range : numpy.ndarray
            Range of values to be considered as possible number of clusters.
        to_plot : bool, optional
            Flag indicating whether the plot of the indices and the optimal number of clusters should be generated,
             by default False.

    Returns
    -------
        int
            Optimal number of clusters.
    """

    if method in {'silhouette', 'elbow', 'calinski_harabasz'}:
        index = np.argmax(indices)
    elif method in {'wb_index', 'xu_index'}:
        index = np.argmin(indices)
    elif method == 'hartigan':
        index = np.where(indices < 10)[0]
        if index.shape[0] == 0:
            index = np.nan
            return index, index
        else:
            index = index[0]
    else:
        print('undefined')
        return -1, -1
    if to_plot:
        fig, axs = plt.subplots(1, 1, figsize=(24, 16))
        axs.plot(k_range, indices)
        axs.axvline(k_range[index], color='green')
        axs.grid()
        axs.set_xticks(k_range)
        axs.axhline(10 * (method == 'hartigan'), color='red')
        axs.set_title(f'The {method} index')
        axs.set_xlabel(xlabel='Number of clusters')
        axs.ylim([np.min(indices[np.where(indices > -np.inf)]), np.max(indices[np.where(indices < np.max)])])
        plt.show()
    return k_range[index], index
