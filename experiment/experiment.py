import numpy as np
#from joblib import Parallel


class AlgorithmPipeline(object):
    def __init__(self, data: np.ndarray, algorithm, init_method: str = 'maxmin_min'):
        self.data = data
        self.algorithm = algorithm(self.data)
        self.init_method = init_method

    def run(self, k_range: np.ndarray = np.arange(1,21,1), exec_number: int = 10, max_iter: int = 70):
        labels = np.zeros((k_range.shape[0], exec_number, self.data.shape[0]))
        for j, k in enumerate(sorted(k_range)):
            for i in range(exec_number):
                labels[j, i], _ = self.algorithm.fit(k=k, init_method=self.init_method, max_iter=max_iter)

        return labels

    # def parallel_run(self, k_range: np.ndarray = np.arange(1,21,1), exec_number: int = 10,
    #                  max_iter: int = 70, n_jobs: int = 4):
    #     labels = np.zeros((k_range.shape[0], exec_number, self.data.shape[0]))
    #     with Parallel(n_jobs=n_jobs) as parallel:




