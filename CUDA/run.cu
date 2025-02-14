#include "testCuda.cuh"
#include <chrono>

// calculates the time from the given start time to now, resets the start time to now, then returns as seconds the time elapsed.
long double calculateTime (std::chrono::time_point<std::chrono::high_resolution_clock>* start) {
    auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - *start);
    *start = std::chrono::high_resolution_clock::now();
    return (long double)end.count()/1000000.0;
}

extern "C" __declspec(dllexport)
void runKernel (double* vec1, int* size1, double* vec2, int* size2, int* k) {
    auto start = std::chrono::high_resolution_clock::now();
    run(vec1, *size1, vec2, *size2, *k);
	cudaDeviceSynchronize();
    std::cout << "Time Elapsed: " << calculateTime(&start) << std::endl;


	fflush(stdout);
}
