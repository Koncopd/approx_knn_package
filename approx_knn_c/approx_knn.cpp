
#include <iostream>
#include <thread>

#include "annoylib.h"
#include "kissrandom.h"

extern "C"
{

#ifdef _WIN32
  __declspec(dllexport)
#endif

void get_distances_and_neighbors(double* X, int N, int D, int* knn, double* dists, int K, int num_trees, int search_k) {

		AnnoyIndex<int, double, Euclidean, Kiss32Random> tree = AnnoyIndex<int, double, Euclidean, Kiss32Random>(D);

    for(int i=0; i<N; ++i){
      double *vec = new double[D];

      for(int z=0; z<D; ++z){
        vec[z] = X[i*D+z];
			}
			tree.add_item(i, vec);

      delete[] vec;
		}
		tree.build(num_trees);

		//Check if it returns enough neighbors
		std::vector<int> closest;
		std::vector<double> closest_distances;
		for (int n = 0; n < 100; n++){
			tree.get_nns_by_item(n, K+1, search_k, &closest, &closest_distances);
			unsigned int neighbors_count = closest.size();
			if (neighbors_count < K+1 ) {
        std::cout<<"Requesting "<<K<<" neighbors, but ANNOY is only giving us "<<neighbors_count<<". Please increase search_k"<<std::endl;
				return;
			}
		}

		const size_t nthreads = std::thread::hardware_concurrency();

    std::cout<<"Parallel ("<<nthreads<<" threads)"<<std::endl;
    std::vector<std::thread> threads(nthreads);
    for(int t = 0;t<nthreads;t++)
    {
      threads[t] = std::thread(std::bind(
        [&](const int bi, const int ei, const int t)
        {
          // loop over all items
          for(int n = bi;n<ei;n++)
          {

            // Find nearest neighbors
            std::vector<int> closest;
            std::vector<double> closest_distances;
            tree.get_nns_by_item(n, K+1, search_k, &closest, &closest_distances);

            for (int m = 0; m < K; m++) {
              knn[n*K + m] = closest[m + 1];
              dists[n*K + m] = closest_distances[m + 1];
            }

            closest.clear();
            closest_distances.clear();

          }
        },t*N/nthreads,(t+1)==nthreads?N:(t+1)*N/nthreads,t));
      }
      std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});

	}

}
