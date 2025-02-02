#include "testCuda.cuh"

void callKernel(int* vec1, int* vec2, int* vec3, int size) {
	run(vec1, vec2, vec3, size);
}

extern "C" __declspec(dllexport)
void runKernel (int* vec1, int* vec2, int* res, int* size) {
	callKernel(vec1, vec2, res, *size);
	cudaDeviceSynchronize();

	for (int a = 0; a < *size; a++) {
		printf("%i ", res[a]);
	}

	fflush(stdout);
}
