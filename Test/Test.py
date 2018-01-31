import time
import numpy as np
from approx_knn_c import get_dists_and_knn
from scanpy.data_structs.data_graph import get_distance_matrix_and_neighbors, get_sparse_distance_matrix

def approx_get_distance_matrix_and_neighbors(X, k, n_jobs=1):
    dists, knn = get_dists_and_knn(X, k-1, 120, 2500)
    dists = dists**2
    Dsq = get_sparse_distance_matrix(knn, dists, X.shape[0], k)
    return Dsq, knn, dists

X = np.random.rand(30000, 2000)

print('Matrix shape', X.shape)

t0 = time.time()
_, knn, dists = get_distance_matrix_and_neighbors(X, 10, n_jobs=1)
t1 = time.time()
print('get_distance_matrix_and_neighbors', t1 - t0, 'seconds')

t0 = time.time()
_, knn_a, dists_a = approx_get_distance_matrix_and_neighbors(X, 10)
t1 = time.time()
print('approx_get_distance_matrix_and_neighbors', t1 - t0, 'seconds')

total = 0

for i, row in enumerate(knn):
    total+=np.in1d(knn_a[i], row).sum()

total/=knn.shape[0]
total/=knn.shape[1]
print('Average precision is', np.ceil(total*100), '%')
