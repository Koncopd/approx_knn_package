import time
import numpy as np
import nmslib, scipy
import utils as u
import multiprocessing as mt

num_jobs = mt.cpu_count()

X = np.random.rand(30000, 2000)

print('Matrix shape', X.shape)

t0 = time.time()
_, knn, dists = u.get_distance_matrix_and_neighbors(X, 10, n_jobs=1)
t1 = time.time()
print('get_distance_matrix_and_neighbors', t1 - t0, 'seconds')

t0 = time.time()
_, knn_a, dists_a = u.approx_annoy_knn(X, 10, num_jobs)
t1 = time.time()
print('approx_annoy_knn', t1 - t0, 'seconds')

t0 = time.time()
_, knn_n, dists_n = u.approx_knn_nmslib(X, 10, n_jobs=num_jobs)
t1 = time.time()
print('knn_nmslib', t1 - t0, 'seconds')

total = 0

for i, row in enumerate(knn):
    total+=np.in1d(knn_a[i], row).sum()

total/=knn.shape[0]
total/=knn.shape[1]
print('Annoy average precision for (num_trees=120, search_k=2500) is', np.ceil(total*100), '%')

total = 0

for i, row in enumerate(knn):
    total+=np.in1d(knn_n[i], row).sum()

total/=knn.shape[0]
total/=knn.shape[1]
print('Nmslib average precision is', np.ceil(total*100), '%')
