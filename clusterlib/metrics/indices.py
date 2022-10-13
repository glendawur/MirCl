import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# input: data X and labels Y

def wss(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
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
def wss_axis(Y: np.ndarray, X: np.ndarray, method: str = 'conventional'):
    return wss(X=X, Y=Y, method=method)


def wss_matrix(X: np.ndarray, L: np.ndarray = None, method: str = 'conventional'):
    inertia = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        inertia[i] = np.apply_along_axis(wss_axis, 1, L[i], X, method).reshape(-1)
    return inertia


def bss(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
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


def bss_axis(Y: np.ndarray, X: np.ndarray, method: str = 'conventional'):
    return bss(X=X, Y=Y, method=method)


def bss_matrix(X: np.ndarray, L: np.ndarray = None, method: str = 'conventional'):
    inertia = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        inertia[i] = np.apply_along_axis(bss_axis, 1, L[i], X, method).reshape(-1)
    return inertia


def calinski_harabasz(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None, method: str = 'conventional'):
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


def count(X: np.ndarray):
    return np.unique(X).shape[0]


def calinski_harabasz_matrix(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, SSB: np.ndarray, aggregation=np.mean):
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return (SSB / (classes - 1)) / (SSW / (X.shape[0] - classes))
    else:
        aggregation((SSB / (classes - 1)) / (SSW / (X.shape[0] - classes)),
                    axis=1)


def xu_index(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None):
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
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return X.shape[1] * np.log(np.sqrt(SSW / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(classes)
    else:
        # return X.shape[1] * np.log(np.sqrt(aggregation(SSW, axis=1) / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(
        #    aggregation(classes, axis=1))
        return aggregation(X.shape[1] * np.log(np.sqrt(SSW / (X.shape[1] * (X.shape[0] ** 2)))) + np.log(classes),
                           axis=1)


def wb_index(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None):
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
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    if aggregation is None:
        return classes * SSW / SSB
    else:
        # return aggregation(classes, axis=1) * aggregation(SSW, axis=1) / aggregation(SSB, axis=1)
        return aggregation(classes * SSW / SSB,
                           axis=1)


def silhouette(X: np.ndarray, Y: np.ndarray = None, centers: np.ndarray = None,
               result: str = 'mean'):
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
    silhouettes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        silhouettes[i] = np.apply_along_axis(silhouette_axis, 1, L[i], X, method).reshape(-1)
    if aggregation is None:
        return silhouettes
    else:
        return aggregation(silhouettes, axis=1)


def silhouette_wss(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, method: str = 'mean', aggregation=np.argmin):
    assert aggregation is not None
    chosen_partitions = L[(np.arange(0, L.shape[0]), aggregation(SSW, axis=1))]
    silhouettes = np.zeros((L.shape[0],))
    for i in range(L.shape[0]):
        silhouettes[i] = silhouette(X, chosen_partitions[i])
    return silhouettes


# NOTE: add parameter method: 'adjusting' (not implemented) and 'fixed' () version
def elbow(SSW: np.ndarray, levels: (int, int) = (1, 1), aggregation=np.mean):
    # if method == 'fixed':
    aggregated = aggregation(SSW, axis=1)
    indices = np.full((SSW.shape[0],), -np.inf)
    frac = (aggregated[:-(levels[0] + levels[1])] - aggregated[levels[0]:-levels[1]]) / (
            aggregated[levels[0]:-levels[1]] - aggregated[(levels[0] + levels[1]):])
    indices[levels[0]:-levels[1]] = frac
    return indices


def hartigan(X: np.ndarray, L: np.ndarray, SSW: np.ndarray, aggregation=np.mean):
    aggregated = aggregation(SSW, axis=1)
    classes = np.zeros((L.shape[0], L.shape[1]))
    for i in range(L.shape[0]):
        classes[i] = np.apply_along_axis(count, 1, L[i])
    num_classes = aggregation(classes, axis=1)
    indices = np.full((SSW.shape[0],), np.inf)
    indices[:-1] = (aggregated[:-1] / aggregated[1:] - 1) * (X.shape[0] - num_classes[:-1] - 1)
    return indices


def find_optimal(indices: np.ndarray, method: str, k_range: np.ndarray, to_plot: bool = False):
    if method in {'silhouette', 'elbow', 'calinski_harabasz'}:
        index = np.argmax(indices)
    elif method in {'wb_index', 'xu_index'}:
        index = np.argmin(indices)
    elif method == 'hartigan':
        index = indices[np.where(indices < 10)]
        if index.shape[0] == 0:
            index = np.nan
        else:
            if index.shape[0] > 1:
                index = index[0]
            else:
                index = int(index)
    else:
        print('undefined')
        return -1
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
