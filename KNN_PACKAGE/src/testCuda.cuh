#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <thrust/sort.h>
#include <chrono>

#define PROFILE false
#define threadsPerBlock 128

auto profileClock = std::chrono::high_resolution_clock::now();
long double runTime = 0;

// calculates the time from the given start time to now, resets the start time to now, then returns as seconds the time elapsed.
long double calculateTime () {
    auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - profileClock);
    profileClock = std::chrono::high_resolution_clock::now();
    long double val = (long double)end.count()/1000000.0;
    runTime += val;
    return val;
}

// code for the function below was provided by this stack overflow article: https://stackoverflow.com/questions/12200486/how-to-remove-duplicates-from-unsorted-stdvector-while-keeping-the-original-or
// by a user named Yury.
template<typename T>
size_t removeDuplicates(std::vector<T>& vec) {
    std::set<T> seen;

    auto newEnd = std::remove_if(vec.begin(), vec.end(), [&seen](const T& value) {
        if (seen.find(value) != std::end(seen))
            return true;

        seen.insert(value);
        return false;
    });

    vec.erase(newEnd, vec.end());

    return vec.size();
}


__global__ void gpukNN (double* x, int size, int k, float* testResult) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    // block memory for reduction
    __shared__ float* blockResult;
    if (threadIdx.x == 0) {
        blockResult = (float*)malloc(sizeof(float));
        *blockResult = 0;
    }

    // knn calculation
    if (i < size) {
        // distance vector
        int pos = i * size;
        double tempVal = x[pos+i];
        for (int a = 0; a < size; a++) {
            x[pos+a] = abs(tempVal-x[pos+a]);
        }

        //sort
        thrust::sort(thrust::seq, &x[pos], &x[pos+size]);

        atomicAdd(blockResult, (float)(k / (size * 2 * x[pos+k])));
    }
    __syncthreads();

    // reduce
    if (threadIdx.x == 0) {
        atomicAdd(testResult, *blockResult);
        free(blockResult);
    }
}

__global__ void multiWrite (double* memoryLocation, double* values, int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        memcpy(&memoryLocation[i*size], values, sizeof(double)*size);
    }
}

float kNN (std::vector<double> x, int k) {
    int N = x.size();

    if(k > N - 1) {
        k = N - 1;
    }

    // slicing into different chunks in the future maybe, if too big?

    #if PROFILE
        std::cout << std::fixed <<"\tKNN Pre-Malloc: " << calculateTime() << std::endl;
    #endif

    // maybe allocate early on at max size, and just use sections?
    double* devMatrixX;
    cudaMalloc(&devMatrixX, sizeof(double) * N * N);

    // vector to copy acrossed devMatrixX
    double* devX;
    cudaMalloc(&devX, sizeof(double) * N);
    cudaMemcpy(devX, x.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    #if PROFILE
        std::cout << std::fixed <<"\tKNN Memcpy: " << calculateTime() << std::endl;
    #endif

    // have the kernel threads call memcpy N times
    multiWrite<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(devMatrixX, devX, N);

    #if PROFILE
        std::cout << std::fixed <<"\tKNN Malloc: " << calculateTime() << std::endl;
    #endif

    // variable to store the reduced result
    float* devResult;
    cudaMalloc(&devResult, sizeof(float));

    // do kNN calculation
    gpukNN<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(devMatrixX, N, k, devResult);

    #if PROFILE
        std::cout << std::fixed << "\tKNN kernel call: " << calculateTime() << std::endl;
    #endif


    // grab the result from the device.
    float result = 0;
    cudaMemcpy(&result, devResult, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devMatrixX);
    cudaFree(devResult);
    cudaFree(devX);


    return result / N;
}

double run (double* data_x, int size1, double* data_y, int size2, int k) {
    // create the yval vector where yval is all singleton values of data_y
    std::vector<double> yval{data_y, data_y + size2};
    removeDuplicates(yval);

    double result = 0;
    // 2gb max heap size
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2147483648);

    // maybe a stream could be useful here?
    for(int i = 0; i < yval.size(); i++) {
        #if PROFILE
            std::cout << std::fixed << "Iteration " << i << " START:" << calculateTime() << std::endl;
        #endif

        std::vector<double> x{};
        //equality vector
        for (int a = 0; a < size1; a++) {
            if (data_y[a] == yval[i]) {
                x.push_back(data_x[a]);
            }
        }

        // calculate the information energy
        float IE = kNN(x, k);

        // calculate the weight and add it to the result.
        double weight = (double)x.size() / (double)size1;
        result += IE * weight;

        #if PROFILE
            std::cout << std::fixed << "Iteration " << i << " END:" << calculateTime() << std::endl;
        #endif
    }
    std::cout << "Final Result: " << result << std::endl;
    return result;
}