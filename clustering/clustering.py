import numpy as np
from scipy.spatial.distance import cdist
from miscellaneous import sse

class Kmeans(object):
    """

    """

    def __init__(self, data: np.ndarray):
        """
        Initialization function

        @param data: ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions

        """
        self.data = data

    def centers_initialization(self, k: int, method: str):
        """
        INPUT:
        *  k - int,  the number of clusters
        *  method -  str, 'maxmin_min' for MaxMin by Minimal Distance initialization, 'maxmin_sum' for MaxMin by Minimal Sum
        OUTPUT:
        *  init_centers - ndarray (k, m), data matrix, where rows are the initialized centers
        """

        init_centers = np.zeros((k, self.data.shape[1]))
        # 'maxmin_min
        if method == 'maxmin_min':
            for i in range(k):
                if i == 0:
                    cen_i = np.random.randint(0, self.data.shape[0])
                else:
                    cen_i = np.min(cdist(self.data, init_centers[:i]), axis=1).argmax()
                init_centers[i] = self.data[cen_i]
        # 'maxmin_sum'
        elif method == 'maxmin_sum':
            temp = []
            for i in range(k):
                if i == 0:
                    cen_i = np.random.randint(0, self.data.shape[0])
                    init_centers[i] = self.data[cen_i]
                    temp.append(cen_i)
                else:
                    cen_i = np.sum(cdist(init_centers[:i], self.data), axis=0).argmax()
                    if cen_i in temp:
                        cen_i = np.sum(np.absolute(
                            cdist(init_centers[:i], self.data) - np.reshape(
                                np.mean(cdist(init_centers[:i], self.data), axis=0),
                                (1, self.data.shape[0]))), axis=0).argmin()

                    init_centers[i] = self.data[cen_i]
                    temp.append(cen_i)
        # if wrong, then "random"
        else:
            indices = np.random.choice(np.arange(0, self.data.shape[0]), k, False)
            init_centers = self.data[indices, :]
        return init_centers

    def fit(self, k: int, init_centers: np.ndarray = None,
            init_method: str = 'maxmin_min', max_iter: int = 50, save_history: bool = False):
        """
        @param k: integer,
        @param init_centers: ndarray
        @param init_method: string
        @param max_iter: int
        @param save_history: boolean

        @return labels: ndarray
        @return centers: ndarray
        @return history: (optional) ndarray
        """
        assert k >= 1

        # define number observations
        n = self.data.shape[0]
        # define number of dimensions
        m = self.data.shape[1]

        if k < 2:
            return np.zeros(self.data.shape[0]), list(np.mean(self.data, axis=0))

        # Check centers
        if init_centers is not None:
            assert np.array(init_centers).shape[0] == k
            assert np.array(init_centers).shape[1] == m
        else:
            init_centers = self.centers_initialization(k, init_method)

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
            distances = cdist(self.data, old_centers)

            assert distances.shape[0] == n
            assert distances.shape[1] == k

            labels = distances.argmin(axis=1)
            assert len(labels) == n

            if save_history:
                history.append(labels)

            for j in range(k):
                new_centers.append(np.mean(self.data[np.where(labels == j)], axis=0))

            if not np.all(np.array(new_centers) == np.array(old_centers)):
                old_centers = new_centers
            else:
                break
            if iteration > max_iter:
                break

        centers = new_centers

        # output
        if save_history:
            return labels.astype(int), centers, history
        else:
            return labels.astype(int), centers


class RandomSwap(object):
    def __init__(self, data: np.ndarray):
        """
        Initialization function

        @param data: ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions

        """
        self.data = data
        self.kmeans = Kmeans(data)

    def fit(self, k: int, max_iter: int = 70, init_centers: np.ndarray = None,
            init_method: str = 'maxmin_min', max_kmeans_iter: int = 50, save_history: bool = False):
        """
        @param k: integer,
        @param max_iter: int
        @param init_centers: ndarray
        @param init_method: string
        @param max_kmeans_iter: int
        @param save_history: boolean

        @return labels: ndarray
        @return centers: ndarray
        @return history: (optional) ndarray
        """
        assert k >= 1

        # define number observations
        n = self.data.shape[0]
        # define number of dimensions
        m = self.data.shape[1]

        if k < 2:
            return np.zeros(self.data.shape[0]), list(np.mean(self.data, axis=0))

        # Check centers
        if init_centers is not None:
            assert np.array(init_centers).shape[0] == k
            assert np.array(init_centers).shape[1] == m
        else:
            init_centers = np.array(self.kmeans.centers_initialization(k, init_method))

        if save_history:
            history = []

        distances = cdist(self.data, init_centers)
        fixed_partition = distances.argmin(axis=1)
        fixed_centers = init_centers
        for j in range(k):
            fixed_centers[j] = np.mean(self.data[np.where(fixed_partition == j)], axis=0)
        for i in range(max_iter):
            new_centers = fixed_centers.copy()
            new_centers[np.random.randint(0, k)] = self.data[np.random.randint(0, n)]
            new_partition, new_centers = self.kmeans.fit(k, init_centers=new_centers, max_iter=max_kmeans_iter)
            # !NB
            if sse(np.array(new_partition), self.data) < sse(np.array(fixed_partition), self.data):
                fixed_partition = new_partition
                fixed_centers = np.array(new_centers)
                if save_history:
                    history.append(fixed_partition)

        if save_history:
            return fixed_partition.astype(int), fixed_centers, history
        else:
            return fixed_partition.astype(int), fixed_centers
