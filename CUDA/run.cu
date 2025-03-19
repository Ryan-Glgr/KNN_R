/**
 * @file: run.cu
 * @brief: Defines the functions for the Cuda implementation
 *
 * @author: Sean Gallaway
 * @date: 3.18.2025
 *
 */

#include "testCuda.cuh"

/*!
 *  \fn void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k)
 *  \brief library function that is exported and expected to be ran in R.
 *  \param vec1 the first column of data, as a pointer
 *  \param size1 the amount of values in the first column, as a pointer
 *  \param vec2 the second column of data, as a pointer
 *  \param size2 the amount of values in the second column, as a pointer
 *  \param k the kth nearest neighbor to grab, as a pointer.
 */
extern "C" __declspec(dllexport)
void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k) {
    run(vec1, *size1, vec2, *size2, *k);
    calculateTime();
    std::cout << "Time Elapsed: " << runTime << std::endl;
	fflush(stdout);
}

// debug values for running this as an executable, not to be used by end users.
double vec1Ex[] = {-3.75, -1.36, -1.06, -1.06, -1.06, -0.92, -0.92, -0.77, -0.48, -0.48, -0.48, -0.48, -0.33, -0.33, -0.33, -0.27, -0.18, -0.18, 0.34, 0.34, 0.54, -1.68, -1.36, -1.36, -1.21, -1.21, -1.06, -1.06, -1.06, -1.06, -0.92, -0.92, -0.92, -0.92, -0.87, -0.77, -0.77, -0.63, -0.63, -0.61, -0.48, -0.48, -0.48, -0.48, -0.48, -0.43, -0.33, -0.33, -0.33, -0.33};
double vec2Ex[] = {-1.5329,-1.5329,-0.4946,-0.7332,1.3469,0.1775,0.6584,-0.7332,1.4565,0.1775,0.6584,-0.6441,-1.5329,1.2818,0.6584,0.5767,0.1775,-1.3074,0.1775,0.6584,0.1775,-0.2514,-0.2514,0.6584,-1.5329,-0.045,-0.2514,0.5767,-1.5329,0.1775,0.6088,0.6584,-1.3074,0.1775,-0.2514,-0.7332,-1.5329,0.6584,1.4565,-0.2514,1.4565,-1.5329,-0.1091,-0.2514,0.1775,1.4565,0.6584,-0.7332,-1.5329,-0.2514};

// debug main() for running this as an executable. not to be ran by end users.
int main () {
    int k = 5000;
    int size = 50;
    runKernel(&vec1Ex[0], &size, &vec2Ex[0], &size, &k);
    return 0;
}
