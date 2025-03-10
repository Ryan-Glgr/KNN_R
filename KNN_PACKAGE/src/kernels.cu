// cudaKernels.cu
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <cmath>
#include <cstring>

extern "C" {

    // Device helper: atomic addition for doubles
    __device__ double atomicDoubleAdd(double* address, double val) {
                                    unsigned long long int* address_as_ull = (unsigned long long int*)address;
                                    unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    // Kernel: Computes a distance vector from each element in the row.
    __global__ void distanceVectorize(double* x, const int size) {
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size) {
            int pos = i * size;
            double tempVal = x[pos+i];
            for (int a = 0; a < size; a++) {
                x[pos+a] = fabs(tempVal - x[pos+a]);
            }
        }
    }

    // Kernel: Performs a k-nearest neighbors reduction.
    __global__ void gpukNN(double* x, const int size, const int k,
                        double* kernelResult, const double weight) {
        
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
        // For simplicity, using shared memory allocation via extern (or dynamic allocation) is omitted.
        // A real implementation may need a different reduction strategy.
        __shared__ double blockResult;
        if (threadIdx.x == 0) {
            blockResult = 0;
        }
        __syncthreads();
        
        if (i < size) {
            int pos = i * size;
            // Sort using thrust on device memory (ensure proper thrust support in PTX builds)
            thrust::sort(thrust::seq, &x[pos], &x[pos+size]);
            atomicDoubleAdd(&blockResult, (k / (size * 2 * x[pos+k])));
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            atomicDoubleAdd(kernelResult, blockResult / size * weight);
        }
    }

    // Kernel: Writes a vector repeatedly into a matrix.
    __global__ void multiWrite(double* memoryLocation, const double* values, const int size) {
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size) {
            memcpy(&memoryLocation[i*size], values, sizeof(double)*size);
        }
    }
}
