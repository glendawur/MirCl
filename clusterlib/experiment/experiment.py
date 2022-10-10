import os
import numpy as np

class AlgorithmPipeline(object):
    def __init__(self, data: np.ndarray, algorithm, init_method: str = 'maxmin_min'):
        self.data = data
        self.algorithm = algorithm(self.data)
        self.init_method = init_method

    def run(self, k_range: np.ndarray = np.arange(1,21,1), exec_number: int = 10, max_iter: int = 70,
            iter_count: bool = False):
        labels = np.zeros((k_range.shape[0], exec_number, self.data.shape[0]))
        exec_time = np.zeros((k_range.shape[0], exec_number))
        iter_counter = np.zeros((k_range.shape[0], exec_number))
        for j, k in enumerate(sorted(k_range)):
            for i in range(exec_number):
                labels[j, i], _, meta = self.algorithm.fit(k=k, init_method=self.init_method, max_iter=max_iter)
                exec_time[j, i] = meta[-2]
                iter_counter[j, i] = meta[-1]
        if iter_count:
            return labels, exec_time, iter_counter
        else:
            return labels, exec_time

    # def parallel_run(self, k_range: np.ndarray = np.arange(1,21,1), exec_number: int = 10,
    #                  max_iter: int = 70, n_jobs: int = 4):
    #     labels = np.zeros((k_range.shape[0], exec_number, self.data.shape[0]))
    #     with Parallel(n_jobs=n_jobs) as parallel:


def save_partition(file: np.ndarray, filename: str, path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), file)


