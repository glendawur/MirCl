import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .miscellaneous import get_1d_metric, calculate_inertia
from .anomalous_clustering import AnomalousPatterns
from .kmeans import KMeans
"""
from miscellaneous import get_1d_metric, calculate_inertia
from anomalous_clustering import AnomalousPatterns
from kmeans import KMeans
"""
import time


def silhouette(X: np.ndarray, Y: np.ndarray, result: str = 'ts', plot: bool = False):
    """
    ATTENTION, for the average silhouette width of all observation use silhouette_mean() function

    INPUT:

    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * Y - ndarray (n, ), array with partition of the data set
    * result - str, the method for the silhouette metric

        * 'ts' - float (default), the average silhouette width of all observations in data set
        * 'as' - ndarray (k, ),  average silhouette width for each cluster
        * 'acs' - float, average silhouette width of all clusters (average of average values)
        * 's' - ndarray (n, ), array with the silhouette width value of each observation

    * plot - bool, if true, than plot silhouette width graph

    OUTPUT:

    * _ - ndarray or float, result depending on the 'result' parameter


    """

    start = time.process_time()

    unique, count = np.unique(Y, return_counts=True)
    if unique.shape[0] == 1:
        return -1, 0.0
    else:
        # matrix, where True if i and j are in the same cluster and i != j
        matrix1 = np.tile(Y, (Y.shape[0], 1)) == np.tile(Y, (Y.shape[0], 1)).T
        np.fill_diagonal(matrix1, False)
        distances = cdist(X, X)

        # array of a(i)
        a = np.sum(distances * matrix1, axis=1) / np.sum(matrix1, axis=1)

        b = np.zeros(Y.shape[0])
        for obs in range(Y.shape[0]):
            distance = np.zeros(unique.shape[0])
            for label in range(unique.shape[0]):
                distance[label] = np.sum(distances[obs, np.where(Y == label)], axis=1) / count[label]
            distance[int(Y[obs])] = np.inf
            b[obs] = np.min(distance)

        silhouette = (b - a) / np.max([a, b], axis=0)

        stop = time.process_time()
        execution_time = stop - start

        if plot:
            temp = []
            for i in np.unique(Y):
                tp = silhouette[np.where(Y == i)]
                tp.sort()
                for j in tp:
                    temp.append(j)
            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax.barh(np.arange(Y.shape[0]), np.array(temp), align='center')
            ax.set_title('Silhouette')

        if result == 'as':
            cluster_slh = {}
            for i in unique:
                cluster_slh[i] = np.mean(silhouette[np.where(Y == i)])
            return cluster_slh, execution_time

        elif result == 'acs':
            acs = 0
            for i in unique:
                acs += np.mean(silhouette[np.where(Y == i)])
            return acs / unique.shape[0], execution_time

        elif result == 'ts':
            return np.mean(silhouette), execution_time

        elif result == 's':
            return silhouette, execution_time


def silhouette_mean(X: np.ndarray, Y: np.ndarray):
    """
    Faster function to calculate silhouette width of the data set

    INPUT:

    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * Y - ndarray (n, ), array with partition of the data set

    OUTPUT:

    * _ - float, the mean value of silhouette width

    """

    start = time.process_time()

    unique, count = np.unique(Y, return_counts=True)
    if unique.shape[0] == 1:
        stop = time.process_time()
        execution_time = stop - start

        return -1, execution_time
    else:
        # matrix, where True if i and j are in the same cluster and i != j
        matrix1 = np.tile(Y, (Y.shape[0], 1)) == np.tile(Y, (Y.shape[0], 1)).T
        np.fill_diagonal(matrix1, False)
        distances = cdist(X, X)

        # array of a(i)
        a = np.sum(distances * matrix1, axis=1) / np.sum(matrix1, axis=1)

        b = np.zeros(Y.shape[0])
        for obs in range(Y.shape[0]):
            distance = np.zeros(unique.shape[0])
            for label in range(unique.shape[0]):
                distance[label] = np.sum(distances[obs, np.where(Y == unique[label])], axis=1) / count[label]
            distance[int(Y[obs])] = np.inf
            b[obs] = np.min(distance)
        stop = time.process_time()
        execution_time = stop - start

        return np.mean((b - a) / np.max([a, b], axis=0)), execution_time


def levene_div(X: np.ndarray, Y: np.ndarray):
    u, c = np.unique(Y, return_counts=True)
    cls_centers = np.zeros((u.shape[0], X.shape[1]))
    CD = 0.0

    ### Calculate local diversion
    for cl, sz in zip(u, c):
        cluster = X[np.where(Y == cl)]
        cls_centers[int(cl)] = cluster.mean(axis=0)
        CD += (1 / sz ** 2) * cdist(cluster, cluster).sum()

    ### Calculate global diversion
    GD = (1 / cls_centers.shape[0] ** 2) * cdist(cls_centers, cls_centers).sum()
    # print('{}, {}'.format(GD, CD))
    return GD - CD

class ChoiceK(object):
    def __init__(self, X: np.ndarray, Y: np.ndarray, L: np.ndarray, range_k: np.ndarray, kmeans_time: float):
        assert L.shape[0] == range_k.shape[0]
        self.L = L
        self.data = X
        self.orig_partition = Y
        self.range_k = range_k
        self.kmeans_time = kmeans_time
        self.wss_matrix, self.wss_time = calculate_inertia(X, L, metric='wss')
        self.wcd_matrix, self.wcd_time = calculate_inertia(X, L, metric='wcd')
        self.wss_min, self.wss_min_labels, self.wss_min_time = get_1d_metric(self.wss_matrix, L, 'min')
        self.wss_mean, self.wss_mean_labels, self.wss_mean_time = get_1d_metric(self.wss_matrix, L, 'mean')
        self.wcd_min, self.wcd_min_labels, self.wcd_min_time = get_1d_metric(self.wcd_matrix, L, 'min')
        self.wcd_mean, self.wcd_mean_labels, self.wcd_mean_time = get_1d_metric(self.wcd_matrix, L, 'mean')

    def silhouette(self, plot: bool = False):
        start = time.process_time()
        silhouettes = np.zeros(self.range_k.shape[0])

        for i in range(self.range_k.shape[0]):
            silhouettes[i], _ = silhouette_mean(self.data, self.wss_min_labels[i])
        k_opt = self.range_k[np.argmax(silhouettes)]
        stop = time.process_time()
        execution_time = (stop - start)+(self.kmeans_time+self.wss_min_time+self.wss_time)

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k, silhouettes)
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Silhouette Width', title='Silhouette')

        return {'k': k_opt, 'time': execution_time}

    def hartigan(self, metric: str = 'wss_mean', plot: bool = False):
        if metric == 'wss_min':
            m = self.wss_min
            add_time = self.kmeans_time+self.wss_time+self.wss_min_time
        elif metric == 'wss_mean':
            m = self.wss_mean
            add_time = self.kmeans_time + self.wss_time + self.wss_mean_time
        elif metric == 'wcd_mean':
            m = self.wcd_mean
            add_time = self.kmeans_time + self.wcd_time + self.wcd_mean_time

        start = time.process_time()
        hartigan = np.zeros(self.range_k.shape[0])

        for i in range(self.range_k.shape[0]):
            if i == 0 or i == (self.range_k.shape[0]-1):
                hartigan[i] = np.inf
            else:
                hartigan[i] = (m[i] / m[i + 1] - 1) * (self.data.shape[0] - (self.range_k[i]) - 1)
        if np.sum(hartigan < 10) == 0:
            k_opt = np.nan
        else:
            k_opt = self.range_k[np.where(hartigan<10)[0][0]]

        stop = time.process_time()
        execution_time = (stop - start) + add_time

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k, hartigan)
            axs.axvline(k_opt, color='green')
            axs.axhline(10, color='red')
            axs.set(xlabel='Number of clusters', ylabel='Hartigan Value H(K)', title='Hartigan Rule')

        return {'k': k_opt, 'time': execution_time}

    def calinski_harabasz(self, plot: bool = False):
        start = time.process_time()
        scatter = np.sum(self.data**2)
        ch = np.zeros(self.range_k.shape[0])

        for i in range(self.range_k.shape[0]):
            if i < 1:
                ch[i] = 0
            else:
                ch[i] = ((scatter - self.wss_min[i])/self.wss_min[i])*((self.data.shape[0] - self.range_k[i])/(self.range_k[i])-1)

        k_opt = self.range_k[np.argmax(ch)]

        stop = time.process_time()
        execution_time = (stop-start)+self.kmeans_time+self.wss_time+self.wss_min_time

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k, ch)
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Calinski-Harabasz Value CH(K)', title='Calinski-Harabasz')

        return {'k': k_opt, 'time': execution_time}

    def elbow(self, method: str = 'adjusting', level: int = 1, plot: bool = False):
        start = time.process_time()
        r = np.zeros(self.range_k.shape[0])
        if method == 'simple':
            for i in range(self.range_k.shape[0]):
                if i+level>self.range_k[-1] or i-level< self.range_k[0]:
                    r[i] = -np.inf
                else:
                    r[i]=(self.wss_min[i - level] - self.wss_min[i]) / (self.wss_min[i] - self.wss_min[i + level])
            limits = (level, -level)
        elif method == 'adjusting':
            for i in range(self.range_k.shape[0]):
                if i - level < 0:
                    min = 0
                else:
                    min = i - level
                if i + level > self.range_k.shape[0]-1:
                    max = self.range_k.shape[0]-1
                else:
                    max = i + level
                r[i]=(self.wss_min[min] - self.wss_min[i]) / (self.wss_min[i] - self.wss_min[max])
                r[0]=-np.inf
                r[-1]=-np.inf
            limits = (1,-2)
        elif method == 'L1L2':
            for i in range(self.range_k.shape[0]):
                if i - 1 < 0:
                    min = 0
                else:
                    min = i - 1
                if i + 2 > self.range_k.shape[0]-1:
                    max = self.range_k.shape[0]-1
                else:
                    max = i + 2
                r[i]=(self.wss_min[min] - self.wss_min[i]) / (self.wss_min[i] - self.wss_min[max])
                r[0] = -np.inf
                r[-1] = -np.inf
            limits = (1, -2)
        elif method == 'L2L1':
            for i in range(self.range_k.shape[0]):
                if i - 2 < 0:
                    min = 0
                else:
                    min = i - 2
                if i + 1 > self.range_k.shape[0]-1:
                    max = self.range_k.shape[0]-1
                else:
                    max = i + 1
                r[i]=(self.wss_min[min] - self.wss_min[i]) / (self.wss_min[i] - self.wss_min[max])
                r[0] = -np.inf
                r[-1] = -np.inf
            limits = (1, -2)
        k_opt = self.range_k[np.argmax(r)]

        stop = time.process_time()
        execution_time = stop - start + self.kmeans_time + self.wss_time + self.wss_min_time

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k[limits[0]:limits[1]], r[limits[0]:limits[1]])
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Curvature', title='Elbow Method')

        return {'k': k_opt, 'time': execution_time, 'index': r}

    def experimental_elbow(self, level: int = 1, plot: bool = False):
        start = time.process_time()
        r = np.zeros(self.range_k.shape[0])
        for i in range(self.range_k.shape[0]):
            if i - level < 0:
                min = 0
            else:
                min = i - level
            if i + level > self.range_k.shape[0] - 1:
                max = self.range_k.shape[0] - 1
            else:
                max = i + level
            r[i] = (self.wss_mean[min] - self.wss_mean[i]) / (self.wss_mean[i] - self.wss_mean[max])
            r[0] = -np.inf
            r[-1] = -np.inf
        limits = (1, -2)

        k_opt = self.range_k[np.argmax(r)]

        stop = time.process_time()
        execution_time = stop - start + self.kmeans_time + self.wss_time + self.wss_mean_time

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k[limits[0]:limits[1]], r[limits[0]:limits[1]])
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Curvature', title='Elbow Method')

        return {'k': k_opt, 'time': execution_time, 'index': r}

    def diversity(self, plot: bool = False):
        start = time.process_time()
        d = np.zeros(self.range_k.shape[0])
        levene_matrix = np.zeros((self.L.shape[0], self.L.shape[1]))
        opt_L = np.zeros((self.L.shape[0], self.L.shape[2]))
        for i in range(self.L.shape[0]):
            for j in range(self.L.shape[1]):
                levene_matrix[i, j] = levene_div(self.data, self.L[i, j])
            opt_j = levene_matrix[i].argmax()
            opt_L[i] = self.L[i, opt_j]
            d[i] = levene_matrix[i, opt_j]

        k_opt = self.range_k[np.argmax(d)]
        opt_labels = opt_L[np.argmax(d)]

        stop = time.process_time()
        execution_time = stop - start + self.kmeans_time

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k, d)
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Diversity Q(K)', title='Levene Diversity')

        return {'k': k_opt, 'time': execution_time, 'partition': opt_labels}
    
    def diversity_wss(self, plot: bool = False):
        start = time.process_time()
        d = np.zeros(self.range_k.shape[0])
        for i in range(self.range_k.shape[0]):
            d[i] = levene_div(self.data, self.wss_min_labels[i])
            
        k_opt = self.range_k[np.argmax(d)]
        opt_labels = self.wss_min_labels[np.argmax(d)]
        
        stop = time.process_time()
        execution_time = stop - start + self.kmeans_time + self.wss_min_time
        
        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(24, 16))
            axs.plot(self.range_k, d)
            axs.axvline(k_opt, color='green')
            axs.set(xlabel='Number of clusters', ylabel='Diversity Q(K)', title='Levene Diversity')

        return {'k': k_opt, 'time': execution_time, 'partition': opt_labels}
            
    
    def anomalous_clustering(self, t: float = 0.94, random = None):
        _, c, execution_time = AnomalousPatterns(self.data, t = t, random_init = random)
        km = KMeans(self.data, k = len(c)).fit(c)
        opt_labels = km.labels

        return {'k': len(c), 'time': execution_time, 'partition': opt_labels}
