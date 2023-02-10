import numpy as np
from scipy.spatial.distance import cdist
import time

from ..miscellaneous import sse, centering


class Kmeans(object):
    """
        A class implementing the for K-means clustering algorithm.

        Parameters
        ----------
            data : np.ndarray
                The data to be clustered, represented as a MxN array, where M is the
                number of samples and N is the number of features.
            metric : str, optional
                The distance metric to use when computing distances between data points.
                Default is 'euclidean'.
            kwargs : optional
                Additional keyword arguments to pass to the underlying KMeans object.

        Attributes
        ----------
            M : int
                The number of samples in the data.
            metric : str
                The distance metric used when computing distances between data points.
            metric_options : dict
                Additional options to pass to the distance metric when computing distances.
            last_call_history : list
                A list of the cluster centers obtained in each iteration of the fit method.
    """

    def __init__(self, data: np.ndarray, metric: str = 'euclidean', **kwargs):
        """
        Initialize the KMeans class.

        Parameters
        ----------
            data : np.ndarray
                The data to be used for clustering.
            metric : str, optional
                The distance metric to be used. Default is 'euclidean'.
            **kwargs : Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.

        Returns
        -------
            None

        """

        self.data = data
        self.M = data.shape[0]
        self.metric = metric
        self.metric_options = {}
        self.last_call_history = list()

    def init_centers(self, k: int, method: str):
        """
        Initialize the centers for clustering.

        Parameters
        ----------
            k : int
                The number of clusters to initialize.
            method : str
                The method of initializing the centers.
                Possible values are 'maxmin' for MaxMin by Minimal Distance initialization,
                'kmeans++' for k-means++ initialization,
            and anything else for random initialization.

        Returns
        -------
            init_centers : numpy.ndarray
                A matrix with the initialized centers. Each row represents a center.
        """

        init_centers = np.zeros((k, self.data.shape[1]))

        # choose the center that is the most distant from other centers already initialized
        # based on the minimum distance
        if method == 'maxmin':
            for i in range(k):
                cen_i = np.random.randint(0, self.data.shape[0]) if i == 0 else np.min(
                    cdist(self.data, init_centers[:i], metric=self.metric, **self.metric_options), axis=1).argmax()
                init_centers[i] = self.data[cen_i]
        # kmeans++ initialization
        elif method == 'kmeans++':
            for i in range(k):
                cen_i = np.random.randint(0, self.data.shape[0]) if i == 0 else np.random.choice(
                    np.arange(0, self.data.shape[0]),
                    p=np.power(
                        cdist(self.data, init_centers[:i], metric=self.metric, **self.metric_options).min(axis=1) /
                        np.linalg.norm(cdist(self.data, init_centers[:i]).min(axis=1)), 2).reshape(-1))
                init_centers[i] = self.data[cen_i]
        # if wrong, then "random"
        else:
            indices = np.random.choice(np.arange(0, self.data.shape[0]), k, False)
            init_centers = self.data[indices, :]
        return init_centers

    def fit(self, k: int, init_centers: np.ndarray = None,
            init_method: str = 'maxmin', max_iter: int = 50):
        """
        Perform K-means clustering on the data.

        Parameters
        ----------
            k : int
                The number of clusters to form as well as the number of centroids to generate.
            init_centers : ndarray, optional
                The initialization method for the centroids. If None, the centroids will be initialized using
                the 'maxmin' method.
            init_method : str, optional
                The method to use for initializing the centroids.
                Must be one of 'maxmin', 'kmeans++', or 'random'.
            max_iter : int, optional
                The maximum number of iterations to perform.

        Returns
        -------
            labels : ndarray
                An integer array of shape (n_samples,) with the indices of the cluster to which each sample belongs.
            centers : ndarray
                An array of shape (k, n_features) containing the cluster centers.
        """
        assert k >= 1, "Number of clusters must be greater than or equal to 1"
        n, m = self.data.shape[0], self.data.shape[1]
        self.last_call_history = list()

        if k < 2:
            return np.zeros(n), np.mean(self.data, axis=0)

        if init_centers is None:
            init_centers = self.init_centers(k, init_method)
        else:
            assert np.array(init_centers).shape == (k, m), "Centers have incorrect shape"

        self.last_call_history.append(np.repeat(self.data.mean(axis=0).reshape(1, -1), repeats=k, axis=0))
        labels = np.zeros(n)
        old_centers = np.array(init_centers)
        iteration = 0

        # convergence process
        while True:

            # init parameters
            new_centers = np.zeros(old_centers.shape)
            iteration += 1
            # distances
            distances = cdist(self.data, old_centers, metric=self.metric, **self.metric_options)

            assert distances.shape[0] == n
            assert distances.shape[1] == k

            labels = one_hot(distances.argmin(axis=1))
            new_centers = (np.matmul(self.data.T, labels) / labels.sum(axis=0)).T
            self.last_call_history.append(new_centers)
            if not np.array_equal(new_centers, old_centers):
                old_centers = new_centers
            else:
                break
            if iteration > max_iter:
                break

        centers = old_centers
        end = time.process_time()
        # output
        return labels.argmax(axis=1).astype(int), centers


class RandomSwap(object):
    """
        A class implementing the Random Swap algorithm for K-means clustering.

        The Random Swap algorithm is a heuristic approach to find the global minimum
        of the sum of squared distances (SSE) between data points and cluster centers.
        It initializes K-means clustering with a specified number of clusters and
        iteratively performs swaps of randomly selected data points between clusters
        to find the best solution.

        Parameters
        ----------
            data : np.ndarray
                The data to be clustered, represented as a MxN array, where M is the
                number of samples and N is the number of features.
            metric : str, optional
                The distance metric to use when computing distances between data points.
                Default is 'euclidean'.
            kwargs : optional
                Additional keyword arguments to pass to the underlying KMeans object.

        Attributes
        ----------
            M : int
                The number of samples in the data.
            metric : str
                The distance metric used when computing distances between data points.
            metric_options : dict
                Additional options to pass to the distance metric when computing distances.
            last_call_history : list
                A list of the cluster centers obtained in each iteration of the fit method.
            km : KMeans
                An instance of the KMeans class, used to perform K-means clustering.

    """

    def __init__(self, data: np.ndarray, metric: str = 'euclidean', **kwargs):
        """
        Initialize the RandowSwap class.

        Parameters
        ----------
            data : np.ndarray
                The data to be used for clustering.
            metric : str, optional
                The distance metric to be used. Default is 'euclidean'.
            **kwargs : Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.

        Returns
        -------
            None
        """

        self.M = data.shape[0]
        self.metric = metric
        self.metric_options = {}
        self.last_call_history = list()
        self.km = Kmeans(data, self.metric, **self.metric_options)

    def fit(self, k: int, init_centers: np.ndarray = None, init_method: str = 'maxmin',
            max_swaps: int = 50, max_convergence_iter: int = 50):
        """
            Perform the Random Swap algorithm to find the best K-means clustering solution.

            Parameters
            ----------
                k : int
                    The number of clusters to use for K-means clustering.
                init_centers : np.ndarray, optional
                    The initial cluster centers to use. If None (default), the cluster centers
                    will be initialized using the specified `init_method`.
                init_method : str, optional
                    The method to use for initializing the cluster centers, if `init_centers`
                    is not specified. Default is 'maxmin'.
                max_swaps : int, optional
                    The maximum number of swaps to perform before ending the algorithm.
                    Default is 50.
                max_convergence_iter : int, optional
                    The maximum number of iterations to use for convergence in each K-means
                    clustering step. Default is 50.

            Returns
            -------
                labels : ndarray
                    An integer array of shape (n_samples,) with the indices of the cluster to which each sample belongs.
                centers : ndarray
                    An array of shape (k, n_features) containing the cluster centers.
        """
        assert k >= 1, "Number of clusters must be greater than or equal to 1"
        n, m = self.km.data.shape[0], self.km.data.shape[1]
        self.last_call_history = list()

        if k < 2:
            return np.zeros(n), np.mean(self.km.data, axis=0)

        if init_centers is None:
            init_centers = self.km.init_centers(k, init_method)
        else:
            assert np.array(init_centers).shape == (k, m), "Centers have incorrect shape"

        self.last_call_history.append(init_centers)
        best_centers = init_centers
        best_labels = None
        best_sse = float('inf')

        counter = 0
        for i in range(max_swaps):
            labels, centers = self.km.fit(k=k, init_centers=init_centers, max_iter=max_convergence_iter)
            if best_sse > sse(self.km.data, centers):
                best_sse = sse(self.km.data, centers)
                best_centers = centers
                best_labels = labels
                self.last_call_history.extend(self.km.last_call_history)
                counter = 0
            elif best_sse == sse(self.km.data, centers):
                counter += 1
                # counter value can be changed
                if counter == 2:
                    break

        return best_labels, best_centers


class AnomalousPatterns(object):
    """
    A class for detecting anomalous patterns in data.

    Parameters
    ----------
        data : np.ndarray, shape (n_samples, n_features)
            The input data to be processed.
        metric : str, optional
            The distance metric used to compare observations. Default is 'euclidean'.
        kwargs : optional
            Additional keyword arguments passed to the distance metric.

    Attributes
    ----------
        data : np.ndarray, shape (n_samples, n_features)
            The input data after centering and normalizing.
        M : int
            The number of observations in the data.
        metric : str
            The distance metric used to compare observations.
        metric_options : dict
            Additional options for the distance metric.
        last_call_history : list
            The history of the algorithm's results for each iteration.

    """

    def __init__(self, data: np.ndarray, metric: str = 'euclidean', **kwargs):
        """
        Initialize the AnomalousPatters class.

        Parameters
        ----------
            data : np.ndarray
                The data to be used for clustering.
            metric : str, optional
                The distance metric to be used. Default is 'euclidean'.
            **kwargs : Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.

        Returns
        -------
            None
        """

        self.data = centering(data, normalize=True)
        self.M = data.shape[0]
        self.metric = metric
        self.metric_options = {}
        self.last_call_history = list()

    def fit(self, max_iter: int = 50):
        """
        Perform clustering on data with anomalous pattern extraction.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations to perform before ending the loop. The default is 50.

        Returns
            -------
             labels : ndarray
                An integer array of shape (n_samples,) with the indices of the anomalous cluster that was extracted.
            centers : ndarray
                An array of shape (k, n_features) containing the anomalous cluster centers.

        """
        n, m = self.data.shape[0], self.data.shape[1]
        self.last_call_history = list()
        partition = np.zeros(self.data.shape[0])

        out_centers = list()
        cluster = 0
        while True:
            cluster += 1
            idx = np.where(partition == 0)
            p_x = self.data[idx]
            old_center = p_x[np.power(np.linalg.norm(p_x, axis=1), 2).argmax()]
            self.last_call_history.append([(cluster, old_center)])

            iteration = 0
            while True:
                iteration += 1
                distances = cdist(p_x, np.array([np.zeros(old_center.shape), old_center]),
                                  self.metric, **self.metric_options)

                labels = distances.argmin(axis=1)
                new_center = p_x[np.where(labels == 1)].mean(axis=0)

                self.last_call_history.append([(cluster, new_center)])

                if not np.array_equal(new_center, old_center):
                    old_center = new_center
                else:
                    break
                if iteration > max_iter:
                    break

            partition[idx[0][np.where(labels == 1)]] = cluster
            out_centers.append(old_center)

            if np.where(partition == 0)[0].shape[0] == 0:
                break

        return partition - 1, np.array(out_centers)
