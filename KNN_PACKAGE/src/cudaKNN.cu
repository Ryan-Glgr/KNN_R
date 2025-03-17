#include "cudaKNN.cuh"
#include "cuda_runtime.h"
/*!
 *  \fn void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k)
 *  \brief library function that is exported and expected to be ran in R.
 *  \param vec1 the first column of data, as a pointer
 *  \param size1 the amount of values in the first column, as a pointer
 *  \param vec2 the second column of data, as a pointer
 *  \param size2 the amount of values in the second column, as a pointer
 *  \param k the kth nearest neighbor to grab, as a pointer.
 */

extern "C" double cudaKNN(double* vec1, int size1, double* vec2, int size2, int k) {
    return run(vec1, size1, vec2, size2, k);
}
