#include "cudaKNN.cuh"
/*!
 *  \fn void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k)
 *  \brief library function that is exported and expected to be ran in R.
 *  \param vec1 the first column of data, as a pointer
 *  \param size1 the amount of values in the first column, as a pointer
 *  \param vec2 the second column of data, as a pointer
 *  \param size2 the amount of values in the second column, as a pointer
 *  \param k the kth nearest neighbor to grab, as a pointer.
 */

// On Windows, we need __declspec(dllexport). otherwise, expand to nothing.
#if defined(_WIN32) || defined(_WIN64)
  #define CUDA_DLL_EXPORT __declspec(dllexport)
#else
  #define CUDA_DLL_EXPORT
#endif

// Use extern "C" to prevent C++ name mangling so we can properly call our function.
extern "C" {
    // This is the function you want to dynamically load:
    CUDA_DLL_EXPORT
    double cudaKNN(double* vec1, int size1, double* vec2, int size2, int k) {
        return run (vec1, size1, vec2, size2, k);
    }
} // end extern "C"