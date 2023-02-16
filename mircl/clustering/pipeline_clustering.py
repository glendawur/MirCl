import os
import numpy as np


class AlgorithmPipeline(object):
    """
    A class to run clustering algorithm multiple times

    Parameters
    ----------
        data : np.ndarray, shape (n_samples, n_features)
            The input data to be processed.
        algorithm : str, optional
            Algorithm to be plugged-in to compute multiple partitions
        init_method: str, optional
            Method to initialize centers for each run
        metric : str, optional
            The distance metric used to compare observations. Default is 'euclidean'.
        **kwargs :
            Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.

    Attributes
    ----------
        data : np.ndarray, shape (n_samples, n_features)
            The input data after centering and normalizing.
        metric : str, optional
            The distance metric used to compare observations. Default is 'euclidean'.
        metric_options : dict
            Additional options for the distance metric.
        algorithm : str, optional
            Algorithm to be plugged-in to compute multiple partitions
        init_method: str, optional
            Method to initialize centers for each run

    """

    def __init__(self, data: np.ndarray, algorithm, init_method: str = 'kmeans++',
                 metric: str = 'euclidean', **kwargs):
        """
        Initialization method of class AlgorithmPipeline

        Parameters
        ----------
            data : np.ndarray, shape (n_samples, n_features)
                The input data to be processed.
            algorithm : str, optional
                Algorithm to be plugged-in to compute multiple partitions
            init_method: str, optional
                Method to initialize centers for each run
            metric : str, optional
                The distance metric used to compare observations. Default is 'euclidean'.
            **kwargs :
                Optional auxiliary arguments to pass to scipy.spatial.distance.cdist to specify metric.

            Returns
            -------
            None
        """
        self.data = data
        self.metric = metric
        self.metric_options = kwargs
        self.algorithm = algorithm(self.data, metric=self.metric, **self.metric_options)
        self.init_method = init_method

    def run(self, k_range: np.ndarray = np.arange(1, 21, 1), exec_number: int = 10, **kwargs):
        """
            Run algorithm for number of times for each given value of k (number of clusters) to compute label matrix for
            further optimal k analysis

            Parameters
            ----------
                k_range : np.ndarray, optional
                    Array with values of number of clusters
                exec_number: int, optional
                    Number of starts for each value of k
                **kwargs:
                    Optional arguments to run the chosen algorithm

            Returns
            -------
                L : np.ndarray (K, E, N)
                    Matrix with labels for each start of algorithm

        """
        labels = np.zeros((k_range.shape[0], exec_number, self.data.shape[0]))

        for j, k in enumerate(sorted(k_range)):
            for i in range(exec_number):
                labels[j, i], _ = self.algorithm.fit(k=k,
                                                     init_method=self.init_method,
                                                     **kwargs)

        return labels
