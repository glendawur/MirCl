import numpy as np
import time
from scipy.spatial import distance_matrix
from .miscellaneous import centering
#from miscellaneous import centering

def filter_clusters(labels: np.ndarray, centers: list, filter_method: str, coefficient: float):
    """
    INPUT:

    * labels - ndarray (n, ), array of partition
    * centers - list with ndarrays with coordinates of centers of extracted clusters
    * filter_method - str, filtering method, expected 
        * 'min' to exclude all clusters with size smaller than coefficient,
        * 'drop' to filter clusters using drop coefficient method (read README for more details)

    OUTPUT:

    * filtered_labels - ndarray (n, ), final partition, where -1 is for the observations from excluded clusters
    * filtered_centers - ndarray (k, m), matrix with coordinates of the centers of the extracted clusters

    """

    # filter the clusters using our criterion of size of cluster t
    unique, counts = np.unique(labels, return_counts=True)

    filtered_centers = []
    filtered_labels = labels
    filter_count = dict(zip(unique, counts))

    if filter_method == 'min':
        for i in filter_count:
            if filter_count[i] < coefficient:
                filtered_labels[np.where(labels == i)] = -1
            else:
                filtered_centers.append(centers[i - 1])
    elif filter_method == 'drop':
        unique, counts = np.unique(labels, return_counts=True)

        filtered_centers = []
        filtered_labels = labels

        filtered_centers = np.array([x for _, x in sorted(zip(counts, centers), key=lambda pair: pair[0])])
        unique = np.array([x for _, x in sorted(zip(counts, unique))])
        counts.sort()
        counts = counts[::-1]
        unique = unique[::-1]
        fil = np.array([True if i == 0 or (((counts[i - 1] - counts[i]) / counts[i - 1]) < coefficient)
                        else False for i in range(counts.shape[0])])
        if np.where(fil == False)[0].shape[0] > 0:
            fil[np.where(fil == False)[0][0]:] = False

        filtered_centers = filtered_centers[np.where(fil == True)]
        filtered_labels[np.where(np.isin(labels, unique[np.where(fil == True)], invert=True))] = -1

    return filtered_labels, filtered_centers


def initialization(data: np.ndarray, random: str = None):
    """
    INPUT:
    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * random - string, 
        * 'random_d' for random choice from observations that are further than 0.75 of the max distance
        * 'random_c' for random choice from 25% of furthest observations
        * 'absolute' - non-random init
    * (OLD)random - bool, initialization method: False to init center of anomalous cluster as the furthest observation, True to init center of anomalous cluster as the random observation out of 20% of the furthest objects

    OUTPUT:
    * centroid - ndarray(m, ), initialized centroid of the cluster
    * labels - ndarray(n, ), the partition after the initialization
    """
    labels = np.zeros(data.shape[0])
    if random == 'random_c':
        number = np.random.randint(1, int(data.shape[0]*0.25-1))
        idx = np.argsort(np.einsum('ij,ij->i', data, data))[-number]
        centroid = data[idx]
        labels[idx] = 1
    elif random == 'random_d':
        dists = np.einsum('ij,ij->i', data, data)
        max_dist = dists[dists.argmax()]
        ids = np.where(dists > (1-0.25)*max_dist)[0]
        idx = ids[np.random.randint(1, ids.shape[0])]
        centroid = data[idx]
        labels[idx] = 1
    elif random == 'absolute' or random == None:
        centroid = data[np.einsum('ij,ij->i', data, data).argmax()]
        labels[np.einsum('ij,ij->i', data, data).argmax()] = 1
    return centroid, labels


# ANOMALOUS PATTERNS BLOCK


def UpdateRule(data: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    INPUT:

    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * centroid - darray (m, ), the center the processed cluster

    OUTPUT:

    * labels - ndarray (n, ), resulting partition after current step
    """
    origin = np.zeros(data.shape[1])

    distances = distance_matrix(data, np.array([origin, centroid]))
    labels = distances.argmin(axis=1)

    return labels


def AP(data: np.ndarray, random: str = None, k_max: int = 50, save_history: bool = False):
    """
    INPUT:

    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * random - bool, if True, new anomalous cluster initialized randomly
    * k_max - (optional) int, the number of maximal iterations of the algorithm
    * save_history - (optional) boolean, if True, output the list history with labels of  each iteration

    OUTPUT:

    * labels - ndarray (n, ), list of labels after the convergence, where n is number of observations
    * center - ndarray (m, ), array with coordinates of the center of the extracted cluster
    * history - (optional) list of length L, that contains L arrays of labels of each iteration

    """
    centroid, labels = initialization(data, random)

    if save_history:
        history = [labels]

    old_c = centroid

    k = 0

    while True:
        k += 1

        labels = UpdateRule(data, old_c)
        new_c = np.mean(data[np.where(labels == 1)], axis=0)

        if np.sum(old_c - new_c) != 0:
            old_c = new_c
            if save_history:
                history.append(labels)
        else:
            break
        if k > k_max:
            break

    centroid = np.mean(data[np.where(labels == 1)], axis=0)
    # print('converged after {} iterations'.format(k))

    if save_history:
        return labels, centroid, history
    else:
        return labels, centroid


def AnomalousPatterns(X: np.ndarray, t: float = 1, filter_method: str = 'drop', g: np.ndarray = None,
                      normalize: bool = False, random_init: str = None, iter_max: int = 50, save_history: bool = False):
    """
    INPUT:

    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * filter_method - str, method for filtering extracted clusters
    * g - ndarray (m, ), array with coordinates of gravity center of the data matrix
    * normalize - bool, if True, than data matrix is also normalized
    * random_init - bool, if True, new anomalous cluster initialized randomly
    * iter_max - int, the upper limit of iterations for the convergence process
    * save_history - bool, if True, than array of partitions of each step is saved

    OTPUT:

    * filtered_labels - ndarray (n, ), array of final partition, where -1 is for the observations from excluded clusters
    * filtered_centers - ndarray (k, m), matrix with coordinates of extracted clusters
    * execution_time - float, time of the algorithm execution
    * history - list with arrays of partitions of each step of the algorithm

    """

    start = time.process_time()

    data_c = centering(X, g, normalize)

    I = data_c

    clusters = []
    centers = []

    if save_history:
        history = []

    while I.shape[0] > 0:
        if save_history:
            cluster, center, cluster_hist = AP(I, k_max=iter_max, save_history=save_history, random=random_init)
            history.append(np.array(cluster_hist))
        else:
            cluster, center = AP(I)

        clusters.append(cluster)
        centers.append(center)

        I = I[np.where(cluster == 0)]

    ####
    for i in range(len(clusters)):
        clusters[i][np.where(clusters[i] == 1)] = i + 1
    for i in range(2, len(clusters) + 1):
        clusters[-i][np.where(clusters[-i] == 0)] = clusters[-i + 1]
    labels = clusters[0]

    filtered_labels, filtered_centers = filter_clusters(labels, centers, filter_method, t)

    stop = time.process_time()
    execution_time = stop - start

    if not save_history:
        return filtered_labels.astype(int), filtered_centers, execution_time
    else:
        return filtered_labels.astype(int), filtered_centers, history, execution_time
    # return labels, centers


# Big Anomalous Clusters One-by-one BLOCK

def CUR(data: np.ndarray, labels: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
   INPUT:

    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * labels - ndarray (n, ), array of partition after previous step
    * centroid - darray (m, ), center the processed cluster

    OUTPUT:

    * new_labels - ndarray (n, ), resulting partition after current step
    """
    cluster_size = len(np.where(labels == 1)[0])

    # failed vectorization
    # temp = np.full((data.shape[0], 2), False)

    # temp[:,0] = cluster_size*np.inner(centroid, centroid)<2*cluster_size*np.einsum('ij, ij->i', np.tile(centroid,
    # (data.shape[0], 1)), data) + np.einsum('ij, ij->i', data, data)

    # temp[:,1] = cluster_size*np.inner(centroid, centroid)<=2*cluster_size*np.einsum('ij, ij->i', np.tile(centroid,
    # (data.shape[0], 1)), data) - np.einsum('ij, ij->i', data, data)

    new_labels = np.full(data.shape[0], 0)
    # new_labels[np.where(labels == 0)] = temp[np.where(labels == 0), 0]
    # new_labels[np.where(labels == 1)] = temp[np.where(labels == 1), 1]

    for i in range(data.shape[0]):
        if labels[i] == 1:
            new_labels[i] = not (cluster_size * np.inner(centroid, centroid) > 2 * cluster_size * np.inner(centroid,
                                                                                                           data[i])
                                 - np.inner(data[i], data[i]))
        else:
            new_labels[i] = cluster_size * np.inner(centroid, centroid) < 2 * cluster_size * np.inner(centroid,
                                                                                                      data[i])\
                            + np.inner(data[i], data[i])

    new_labels = new_labels.astype(int)

    return new_labels


def EXTAN(data: np.ndarray, random: bool = False, k_max: int=50, save_history: bool =False):
    """
    INPUT:

    * data - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * random - bool, if True, new anomalous cluster initialized randomly
    * k_max - (optional) int, the number of maximal iterations of the algorithm
    * save_history - (optional) boolean, if True, output the list history with labels of  each iteration

    OUTPUT:

    * labels - ndarray (n, ), list of labels after the convegance, where n is number of observations
    * center - ndarray (m, ), array with coordinates of the center of the extracted cluster
    * history - (optional) list of length L, that contains L arrays of labels of each iteration

    """
    centroid, cluster = initialization(data, random)

    if save_history:
        history = [cluster]

    old_c = centroid

    k = 0

    while True:
        k += 1

        cluster = CUR(data, cluster, old_c)
        new_c = np.mean(data[np.where(cluster == 1)], axis=0)

        if np.sum(old_c - new_c) != 0:
            old_c = new_c
            if save_history:
                history.append(cluster)
        else:
            break
        # end iteration process if number of iterations is more than parameter
        if k > k_max:
            break

    centroid = np.mean(data[np.where(cluster == 1)], axis=0)

    # print('converged after {} iterations'.format(k))
    if save_history:
        return cluster, centroid, history
    else:
        return cluster, centroid


def BANCO(data: np.ndarray, t: float = 1.0, filter_method: str = 'min', g: np.ndarray = None,
          normalize: bool = False, random_init: bool = False, iter_max: int = 50, save_history: bool = False):
    """
    INPUT:

    * X - ndarray (n, m), data matrix, where n is the number of observations and m is the number of dimensions
    * filter_method - str, method for filtering extracted clusters
    * g - ndarray (m, ), array with coordinates of gravity center of the data matrix
    * normalize - bool, if True, than data matrix is also normalized
    * random_init - bool, if True, new anomalous cluster initialized randomly
    * iter_max - int, the upper limit of iterations for the convergence process
    * save_history - bool, if True, than array of partitions of each step is saved

    OTPUT:

    * filtered_labels - ndarray (n, ), array of final partition, where -1 is for the observations from excluded clusters
    * filtered_centers - ndarray (k, m), matrix with coordinates of extracted clusters
    * execution_time - float, time of the algorithm execution
    * history - list with arrays of partitions of each step of the algorithm

    """
    start = time.process_time()
    # preprocess the data
    data = centering(data=data, g=g, normalize=normalize)

    clusters = []
    centers = []

    I = data

    if save_history:
        history = []

    while I.shape[0] > 0:

        if save_history:
            cluster, center, cluster_hist = EXTAN(I, k_max=iter_max, save_history=save_history, random=random_init)
            history.append(np.array(cluster_hist))
        else:
            cluster, center = EXTAN(I)

        clusters.append(cluster)
        centers.append(center)

        I = I[np.where(cluster == 0)]

    ####
    for i in range(len(clusters)):
        clusters[i][np.where(clusters[i] == 1)] = i + 1
    for i in range(2, len(clusters) + 1):
        clusters[-i][np.where(clusters[-i] == 0)] = clusters[-i + 1]
    labels = clusters[0]

    filtered_labels, filtered_centers = filter_clusters(labels, centers, filter_method, t)

    stop = time.process_time()
    execution_time = stop - start

    if save_history == True:
        return filtered_labels.astype(int), filtered_centers, history, execution_time
    else:
        return filtered_labels.astype(int), filtered_centers, execution_time
