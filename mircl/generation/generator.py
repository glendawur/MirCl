import os
import json
import _pickle as pickle
import numpy as np
import pandas as pd


def generate_size(N: int, n: int, k: int) -> np.ndarray:
    """
    Generate an array of size k that sums up to N and each element is at least n.

    Parameters
    ----------
    N : int
        The total sum of the generated array.
    n : int
        The minimum value of each element in the generated array.
    k : int
        The size of the generated array.

    Returns
    -------
    np.ndarray
        An array of size k that sums up to N and each element is at least n.

    Examples
    --------
    >> generate_size(10, 2, 4)
    array([2, 2, 3, 3])
    """
    G = N - n * k
    dots = np.sort(np.random.randint(0, G, k - 1))
    sizes = np.concatenate([[0], dots, [G]])
    sizes = sizes[1:] - sizes[:-1] + n
    assert sizes.sum() == N
    return sizes.astype(int)


def generate_blobs_set(N: int, M: int, k: int = -1, n: int = -1, a: float = -1, meta: bool = False):
    """Generate a set of points that belongs to `k` clusters in `M` dimensional space.

    Parameters
    ----------
        N : int
            Number of points in the generated set.
        M : int
            Number of dimensions in the generated set.
        k : int, optional
            Number of clusters, by default -1, chosen randomly if -1.
        n : int, optional
            Minimal number of points in each cluster, by default -1, chosen randomly if -1.
        a : float, optional
            Intermix (overlap) ratio of clusters, takes values between 0 and 1.
            Closer to 0 the more overlapped the clusters are. By default -1, chosen randomly if -1.
        meta : bool, optional
            If True, return metadata as well, by default False.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float, Dict[str, Union[int, float, np.ndarray]]]]]
            The generated data set and the labels, and metadata if `meta` is True.

    Raises
    ------
        AssertionError
            If the minimal size of the cluster is not suitable for the size of the whole dataset.
    """
    if k == -1:
        k = np.random.randint(2, 20, 1)[0]
    if n == -1:
        n = ((N / k) * np.random.uniform(0.05, 0.5, 1)).astype(int)[0]
    if a == -1:
        a = np.random.uniform(0.4, 1, 1)[0]

    metadata = {'N': N, 'M': M, 'k': k, 'n': n, 'a': a}
    assert n * k <= N, 'The minimal size of the cluster is not suitable for the size of the whole dataset'

    sizes = generate_size(N, n, k)
    clusters = []
    labels = []

    for cl in range(k):
        cluster = np.zeros((sizes[cl], M))
        center = np.random.uniform(a * (-1), a * 1, M)
        std = np.random.uniform(0.05, 0.1, M)

        for atr in range(M):
            cluster[:, atr] = np.random.normal(0, std[atr], sizes[cl])

        cluster = cluster + center
        key = 'cl{}'.format(cl)
        metadata[key] = {'center': center, 'variances': std, 'size': sizes[cl]}
        clusters.append(cluster)
        labels.append(np.full(sizes[cl], cl))

    X = np.concatenate(clusters, axis=0)
    labels = np.concatenate(labels, axis=0)
    if meta:
        return X, labels, metadata
    else:
        return X, labels


def generate_sphere(n_samples: int = 1000, n_features: int = 2, radius: int = 5, noise: float = 0.05):
    """
    Generates a set of points that lie on the surface of a sphere. The number of points and the number of dimensions
    for the points can be specified. The sphere can also be scaled and perturbed by some noise.

    Parameters
    ----------
    n_samples: int, optional (default=1000)
        Number of points to be generated.
    n_features: int, optional (default=2)
        Number of dimensions for the points.
    radius: int, optional (default=5)
        The radius of the sphere.
    noise: float, optional (default=0.05)
        A factor to perturb the radius of the sphere by some random noise.

    Returns
    -------
    x: array of shape (n_samples, n_features)
        The set of points that lie on the surface of the sphere.

    Example
    -------
    >> x = generate_sphere(n_samples=10, n_features=3)
    >> x.shape
    (10, 3)
    """
    rad = np.random.normal(radius, radius * noise, n_samples)

    thetas = np.zeros((n_samples, n_features - 1))
    thetas[:, 1:] = np.random.uniform(-np.pi / 2, np.pi / 2, (n_samples, n_features - 2))
    thetas[:, 0] = np.random.uniform(0, 2 * np.pi, n_samples)

    cos = np.cos(thetas)[:, ::-1].cumprod(axis=1)[:, ::-1]
    sin = np.sin(thetas)

    x = np.zeros((n_samples, n_features))
    x[:, 0] = rad * cos[:, 0]
    x[:, -1] = rad * sin[:, -1]

    for i in range(1, n_features - 1):
        x[:, i] = rad * cos[:, i] * sin[:, i - 1]

    return x

# def create_generation(N: int, M: np.ndarray, k: np.ndarray, a: np.ndarray, min_size: int,
#                       dataset_number: int, main_path: str):
#     """
#     All generated datasets saved to directory as csv files (subdirectory 'csv') with the last column as the initial partition. Parameters of generation of each dataset are saved to dictionary which is saved as a .txt (subdirectory 'txt') and .pickle (subdirectory 'pickle') files. Finally, total number of generated datasets is printed
#
#     @param N: int, number of observations in the dataset
#     @param M: ndarray, range of values of dimensionality of the dataset
#     @param k: ndarray, range of number of clusters in the dataset
#     @param a: ndarray, range of the values for the coefficient of the intermix of the clusters
#     @param min_size: int, the minimal size of the cluster
#     @param dataset_number: int, number of datasets to generate for each configuration of hyperparameters
#     @param main_path: str, path to save all the generated data to
#
#     """
#     total_num = 0
#
#     path = '{}'.format(main_path)
#
#     if not os.path.exists(path):
#         os.mkdir(path)
#
#     csv_path = os.path.join(path, 'csv')
#     os.mkdir(csv_path)
#     pickle_path = os.path.join(path, 'pickle')
#     os.mkdir(pickle_path)
#     txt_path = os.path.join(path, 'txt')
#     os.mkdir(txt_path)
#
#     for dim in M:
#         for clus_n in k:
#             for intermix in a:
#                 for i in range(dataset_number):
#                     name = 'N{}_M{}_a{}_k{}_id{}'.format(N, dim, intermix, clus_n, i)
#                     X, labels, metadata = generate_set(N=N, M=dim, k=clus_n, n=min_size, a=intermix, meta=True)
#
#                     csv_name = os.path.join(csv_path, name)
#
#                     df = pd.DataFrame(X)
#                     df['labels'] = pd.Categorical(labels)
#                     df.to_csv('{}.csv'.format(csv_name), index=False)
#
#                     txt_name = os.path.join(txt_path, name)
#                     with open('{}_metadata.txt'.format(txt_name), 'w') as file:
#                         file.write(json.dumps(str(metadata)))
#
#                     pickle_name = os.path.join(pickle_path, name)
#                     with open('{}_metadata.pickle'.format(pickle_name), 'wb') as handle:
#                         pickle.dump(metadata, handle)
#
#                     total_num += 1
#
#     print('{} datasets are generated and saved to directory {}'.format(total_num, path))
