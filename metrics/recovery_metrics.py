import numpy as np
from scipy.special import comb
from miscellaneous import pairing_matrix


# from miscellaneous import pairing_matrix
def ari(labels1: np.ndarray, labels2: np.ndarray, result: str = 'ri'):
    """
    INPUT:
    * labels1 - ndarray (n, ), where n is the size of dataset; the array of labels with clustering №1
    * labels2 - ndarray (n, ), where n is the size of dataset; the array of labels with clustering №2
    * result - str, 'ri' for the Rand Index, 'ari' for Adjusted Rand Index
    OUTPUT:
    * RI/ARI - the score of the metric according to the result parameter,
    float from [0,1], value of (adjusted) rand index - similarity index of two partitions
    """
    # calculate contingency matrix with our handmade function
    contingency_matrix = pairing_matrix(labels1, labels2)

    # calculate sum for each row of contingency matrix(clustering №1)
    a = np.sum(contingency_matrix, axis=1)
    # calculate sum for each column of contingency matrix (clustering №2)
    b = np.sum(contingency_matrix, axis=0)

    # calculate auxiliary combination value
    CombN = comb(np.sum(contingency_matrix), 2)

    # calculate auxiliary values
    N_11 = 0.5 * (np.sum(contingency_matrix ** 2) - np.sum(contingency_matrix))
    N_10 = 0.5 * (np.sum(a ** 2) - np.sum(contingency_matrix ** 2))
    N_01 = 0.5 * (np.sum(b ** 2) - np.sum(contingency_matrix ** 2))
    N_00 = CombN - N_11 - N_01 - N_10

    # Calculate Rand Index
    RI = (N_11 + N_00) / (CombN)

    # if variable result = 'ari', calculate Adjusted Rand Index
    if result == 'ari':

        # Calculate auxiliary values for each cluster
        CombA = np.sum(np.array([comb(a_i, 2) for a_i in a]))
        CombB = np.sum(np.array([comb(b_i, 2) for b_i in b]))

        # Calculate auxiliary values of each resulting entry of contingency matrix
        CombAB = 0
        for row in contingency_matrix:
            for val in row:
                CombAB += comb(val, 2)
        return (CombN * CombAB - CombA * CombB) / (0.5 * CombN * (CombA + CombB) - CombA * CombB)
    else:
        return RI


def ami(labels1, labels2, result='mi', perm_max='max'):
    """
    INPUT:
    * labels1 - numpy array (n,), where n is the size of dataset; the array of labels with clustering №1
    * labels2 - numpy array (n,), where n is the size of dataset; the array of labels with clustering №2
    * result - string, variable that define the output value
        * 'mi', returns Mutual Information
        * 'nmi', returns normalized Mutual Information as MI/maximum expected MI
        * 'ami', returns Adjusted Mutual Information
    * perm_max - string, variable that defines permitted maximum value of mutual information
        * 'sqrt', returns the square root of product of two entropies - for clustering №1 and clustering №2
        * 'avg', returns the average (mean) of two entropies - for clustering №1 and clustering №2
        * 'min', returns the smalles entropy value out of two entropies - for clustering №1 and clustering №2
        * 'max', returns the biggest entropy value out of two entropies - for clustering №1 and clustering №2
    OUTPUT:
    * float from [0,1], value of (adjusted/normalized) mutual information - similarity index of two partitions
    """
    contingency_matrix = pairing_matrix(labels1, labels2)

    # calculate the size of dataset
    N = np.sum(contingency_matrix)

    # calculate sum for each row of contingency matrix(clustering №1)
    a = np.sum(contingency_matrix, axis=1)

    # calculate sum for each column of contingency matrix (clustering №2)
    b = np.sum(contingency_matrix, axis=0)

    # calculate entropy for each single event
    # for each cluster in clustering №1
    H_a = - np.sum(np.log2(a / N) * (a / N))

    # for each cluster in clustering №2
    H_b = - np.sum(np.log2(b / N) * (b / N))

    MI = np.sum((contingency_matrix / N) * np.log2((contingency_matrix * N) / np.outer(a.T, b),
                                                   out=np.zeros_like(contingency_matrix),
                                                   where=(contingency_matrix != 0)))

    # return mutual information
    if result == 'mi':
        return MI
    # calculate the permitted maximum value of entropy
    elif result == 'ami' or 'nmi':
        if perm_max == 'sqrt':
            maximum = np.sqrt(H_a * H_b)
        elif perm_max == 'avg':
            maximum = np.mean([H_a, H_b])
        elif perm_max == 'min':
            maximum = np.min([H_a, H_b])
        else:
            maximum = np.max([H_a, H_b])

        # return normalized mutual information, if variable result = 'nmi'
        if result == 'nmi':
            return MI / maximum

        # return adjusted normalized information, if variable result = 'ami'
        else:
            emi = 0
            for a_i in a:
                for b_j in b:

                    # define all possible values of the entry
                    exp_cell = np.arange(np.max([0, int(a_i + b_j - N)]), np.min([a_i, b_j]))  # !!!
                    for n_ij in exp_cell:
                        if n_ij != 0:
                            prob = comb(b_j, n_ij, exact=False) * comb(N - b_j, a_i - n_ij, exact=False) / comb(N, a_i,
                                                                                                                exact=False)
                            emi += (n_ij / N) * (np.log2(n_ij) + np.log2(N) - np.log2(a_i) - np.log2(b_j)) * prob

            AMI = (MI - emi) / (maximum - emi)
            return AMI
    else:
        return MI