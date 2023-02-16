# MirCl

### [Miraculous Clustering]

MirCl is a small package that was initially as code repository
for my bachelor thesis ([Application of Anomalous Clustering 
Methods for Determination оf the Number of Clusters](https://www.hse.ru/en/edu/vkr/469631392)) 
and further research under the supervision.

As of now, this package contains just a few useful tools to
perform clustering analysis:

1. Clustering techniques implementation:
   1. K-Means 
   2. Random Swap K-Means ([2018, Franti](https://link.springer.com/article/10.1186/s40537-018-0122-y))
   3. Anomalous Patterns ([2011, Amorim & Mirkin](https://www.sciencedirect.com/science/article/pii/S0031320311003517))
2. Generating Synthetic Data:
   1. Generator of N-dimensional spheres
   2. Generating a dataset according to ([2020, Taran & Mirkin](https://link.springer.com/article/10.1007/s40685-019-00106-9))
3. Indices to choose the optimal number of clusters:
   1. Analytical Elbow
   2. Hartigan Rule
   3. Calinski-Harabasz
   4. Silhouette Width
   5. Xu index
   6. WB index
4. Metrics to evaluate partitions in supervised way:
   1. Adjusted Rand Index
   2. Normalized/Adjusted Mutual Information

You can find two showcase notebooks in [this folder](showcase/)

![Miraculous Example](/showcase/pics/km_example.gif)

### To-do:

- [] Add stochastic Maxmin initialization
- [] Add more generators of synthetic data
- [] Add jax\numba fast computation of distances
- [] Add batch versions of clustering techniques
- [] Add modifications of Anomalous Patterns algorithm
- [] Add more metrics to evaluate the partition

### Requirements:

* numpy>=1.21.5
* scipy>=1.9.1
* pandas>=1.4.4
* matplotlib>=3.5.2