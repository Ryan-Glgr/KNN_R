#include "testCuda.cuh"


extern "C" __declspec(dllexport)
void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k) {
    run(vec1, *size1, vec2, *size2, *k);
	cudaDeviceSynchronize();
    calculateTime();
    std::cout << "Time Elapsed: " << runTime << std::endl;
    cudaDeviceReset();


	fflush(stdout);
}
