#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(int* vec1, int* vec2, int* vec3, int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	vec3[i] = vec2[i] + vec1[i];
}


void run (int* vec1, int* vec2, int* vec3, int size) {
	int* devVec1;
	int* devVec2;
	int* devVec3;

	cudaMalloc(&devVec1, sizeof(int) * size);
	cudaMalloc(&devVec2, sizeof(int) * size);
	cudaMalloc(&devVec3, sizeof(int) * size);

	cudaMemcpy(devVec1, &vec1[0], sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, &vec2[0], sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec3, 0, sizeof(int) * size, cudaMemcpyHostToDevice);

	kernel <<<1, size >>> (devVec1, devVec2, devVec3, size);

	cudaDeviceSynchronize();
	cudaMemcpy(&vec3[0], devVec3, sizeof(int) * size, cudaMemcpyDeviceToHost);

	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);
}
