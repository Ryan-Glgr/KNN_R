#include "testCuda.cuh"
#include "library.h"
#ifdef HAVE_CUDA
double cudaKNN(double* vec1, int size1, double* vec2, int size2, int K) {
    double r = run(vec1, size1, vec2, size2, K);
	cudaDeviceSynchronize();
    cudaDeviceReset();
    return r;
}
#endif