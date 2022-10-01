import os
import json
import _pickle as pickle
import numpy as np
import pandas as pd


def generate_size(N: int, n: int, k: int) -> np.ndarray:
    """
    INPUT:
    * N - int, number of observations in the dataset
    * n - int, the minimal size of the cluster
    * k - int, number of clusters in dataset
    OUTPUT:
    * sizes - ndarray(k, ), array with sizes of each cluster
    """
    # define the random part of our dataset
    G = N - n * k

    # generate k-1 dots on the 1D linear space between n and N
    dots = np.random.randint(0, G, k - 1)

    # sort results in ascending way
    dots.sort()

    # init temporary arrays
    temp1 = np.zeros(k)
    temp2 = np.full(k, G)

    # change the values of the arrays
    temp1[1:k] = dots
    temp2[0:k - 1] = dots

    # define sizes as the residual of two arrays, each result is the distance between two neighbour points
    sizes = temp2 - temp1

    # add n to each size so as to met the demand of minimal size
    sizes = sizes + n
    assert sizes.sum() == N
    return sizes.astype(int)


def generate_set(N: int, M: int, k: int = -1, n: int = -1, a: float = -1, meta: bool = False):
    """
    @param N: int, number of observations in the dataset
    @param M: int, dimensionality of the dataset
    @param k: int, number of clusters in the dataset
    @param n: int, the minimal size of the cluster
    @param a: float, the coefficient of the intermix of the clusters
    @param meta: bool, if True, parameters of generation of each cluster and each feature are saved


    @return X: ndarray(N, M), data matrix
    @return labels: ndarray(N, ), array of the partition
    """
    # generate parameters randomly in case of empty input
    if k == -1:
        k = np.random.randint(2, 20, 1)[0]
    if n == -1:
        n = ((N / k) * np.random.uniform(0.05, 0.5, 1)).astype(int)[0]
    if a == -1:
        a = np.random.uniform(0.4, 1, 1)[0]

    # dictionary to keep input parameters and generated clusters data
    metadata = {'N': N, 'M': M, 'k': k, 'n': n, 'a': a}

    # assert if the minimal size of cluster is suitable for the size of whole dataset
    # done if input is custom
    assert n * k <= N

    # generate sizes of clusters
    sizes = generate_size(N, n, k)

    # init the lists to keep intermediate results
    clusters = []
    labels = []

    # loop to generate each cluster separately o
    for cl in range(k):

        # generate the matrix to keep generated data
        cluster = np.zeros((sizes[cl], M))

        # generate center of the cluster
        center = np.random.uniform(a * (-1), a * (1), M)

        # generate array of standard deviations for each attribute
        std = np.random.uniform(0.05, 0.1, M)

        # loop to generate array of values for each of attributes
        for atr in range(M):
            # generating the values of the attribute using Normal distribution with pre-generated parameters
            cluster[:, atr] = np.random.normal(0, std[atr], sizes[cl])

        # adding the centering values to the matrix
        cluster = cluster + center

        # creating the key value for metadata dictionary
        key = 'cl{}'.format(cl)

        # saving metadata of the cluster generation to the metadata dictionary
        metadata[key] = {'center': center, 'variances': std, 'size': sizes[cl]}

        # adding generated cluster to list of clusters
        clusters.append(cluster)

        # adding labels to labels list
        labels.append(np.full(sizes[cl], cl))

    # concatenating generated cluster into the single data matrix
    X = np.concatenate(clusters, axis=0)

    # concatenating label arrays into the single label array
    labels = np.concatenate(labels, axis=0)

    # return output
    if meta == True:
        return X, labels, metadata
    else:
        return X, labels


def create_generation(N: int, M: np.ndarray, k: np.ndarray, a: np.ndarray, min_size: int,
                      dataset_number: int, main_path: str):
    """
    All generated datasets saved to directory as csv files (subdirectory 'csv') with the last column as the initial partition. Parameters of generation of each dataset are saved to dictionary which is saved as a .txt (subdirectory 'txt') and .pickle (subdirectory 'pickle') files. Finally, total number of generated datasets is printed

    @param N: int, number of observations in the dataset
    @param M: ndarray, range of values of dimensionality of the dataset
    @param k: ndarray, range of number of clusters in the dataset
    @param a: ndarray, range of the values for the coefficient of the intermix of the clusters
    @param min_size: int, the minimal size of the cluster
    @param dataset_number: int, number of datasets to generate for each configuration of hyperparameters
    @param main_path: str, path to save all the generated data to

    """
    total_num = 0

    path = '{}'.format(main_path)

    if not os.path.exists(path):
        os.mkdir(path)

    csv_path = os.path.join(path, 'csv')
    os.mkdir(csv_path)
    pickle_path = os.path.join(path, 'pickle')
    os.mkdir(pickle_path)
    txt_path = os.path.join(path, 'txt')
    os.mkdir(txt_path)

    for dim in M:
        for clus_n in k:
            for intermix in a:
                for i in range(dataset_number):
                    name = 'N{}_M{}_a{}_k{}_id{}'.format(N, dim, intermix, clus_n, i)
                    X, labels, metadata = generate_set(N=N, M=dim, k=clus_n, n=min_size, a=intermix, meta=True)

                    csv_name = os.path.join(csv_path, name)

                    df = pd.DataFrame(X)
                    df['labels'] = pd.Categorical(labels)
                    df.to_csv('{}.csv'.format(csv_name), index=False)

                    txt_name = os.path.join(txt_path, name)
                    with open('{}_metadata.txt'.format(txt_name), 'w') as file:
                        file.write(json.dumps(str(metadata)))

                    pickle_name = os.path.join(pickle_path, name)
                    with open('{}_metadata.pickle'.format(pickle_name), 'wb') as handle:
                        pickle.dump(metadata, handle)

                    total_num += 1

    print('{} datasets are generated and saved to directory {}'.format(total_num, path))