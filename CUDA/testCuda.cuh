
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <thrust/sort.h>
#include <chrono>

#define PROFILE false
#define threadsPerBlock 128
#define threadCount 1

auto profileClock = std::chrono::high_resolution_clock::now();
long double runTime = 0;


__device__ double* devResult;



__device__ double atomicDoubleAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}



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

__global__ void distanceVectorize (double* x, int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        // distance vector
        int pos = i * size;
        double tempVal = x[pos+i];
        for (int a = 0; a < size; a++) {
            x[pos+a] = abs(tempVal-x[pos+a]);
        }
    }
}


__global__ void gpukNN (double* x, int size, int k, double* testResult, double weight) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    // block memory for reduction
    __shared__ double* blockResult;
    if (threadIdx.x == 0) {
        blockResult = (double*)malloc(sizeof(double));
        *blockResult = 0;
    }

    // knn calculation
    if (i < size) {
        int pos = i * size;
        //sort
        thrust::sort(thrust::seq, &x[pos], &x[pos+size]);
        //
        atomicDoubleAdd(blockResult, (k / (size * 2 * x[pos+k])));
    }
    __syncthreads();

    // reduce
    if (threadIdx.x == 0) {
        atomicDoubleAdd(testResult, *blockResult/size * weight);
        free(blockResult);
    }
}

__global__ void multiWrite (double* memoryLocation, double* values, int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        memcpy(&memoryLocation[i*size], values, sizeof(double)*size);
    }
}

void kNN (std::vector<double> x, int k, int writeLocation, double weight, double* devMatrixX, int* prevMatrixXSize) {
    int N = x.size();

    if(k > N - 1) {
        k = N - 1;
    }

    if (N <= 1) {
        return;
    }

    // slicing into different chunks in the future maybe, if too big?

    #if PROFILE
        std::cout << std::fixed <<"\tKNN Pre-Malloc: " << calculateTime() << std::endl;
    #endif

//    double* devMatrixX;
//    cudaMalloc(&devMatrixX, sizeof(double) * N * N);
    if (*prevMatrixXSize < N*N) {
        cudaFree(devMatrixX);
        cudaMalloc(&devMatrixX, sizeof(double) * N * N);
    }


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
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    #endif

    // do kNN calculation
    distanceVectorize<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(devMatrixX, N);
    #if PROFILE
        std::cout << std::fixed << "\tKNN kernel call: " << calculateTime() << std::endl;
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaDeviceSynchronize();
    #endif

    gpukNN<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(devMatrixX, N, k, &devResult[writeLocation], weight);
    #if PROFILE
        std::cout << std::fixed << "\tKNN kernel call: " << calculateTime() << std::endl;
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
        cudaDeviceSynchronize();
    #endif


    cudaFree(devX);
    #if PROFILE
        std::cout << std::fixed << "\tKNN end: " << calculateTime() << std::endl;
    #endif
}

void threadRun (int index, int k, double* data_x, int size1, double* data_y, std::vector<double>* yval) {
    double* devMatrixX;
    int prevMatrixXSize = yval->size()*yval->size();
    cudaMalloc(&devMatrixX, sizeof(double)*prevMatrixXSize);
    for (int i = index; i < yval->size(); i += threadCount) {
        #if PROFILE
                std::cout << std::fixed << "Iteration " << i << " START:" << calculateTime() << std::endl;
        #endif

        std::vector<double> x{};
        //equality vector
        for (int a = 0; a < size1; a++) {
            if (data_y[a] == yval->at(i)) {
                x.push_back(data_x[a]);
            }
        }

        // calculate the information energy
        kNN(x, k, i, (double) x.size() / (double) size1, devMatrixX, &prevMatrixXSize);
        #if PROFILE
                std::cout << std::fixed << "Iteration " << i << " END:" << calculateTime() << std::endl;
        #endif
    }
    cudaFree(devMatrixX);
}

void run (double* data_x, int size1, double* data_y, int size2, int k) {
    // create the yval vector where yval is all singleton values of data_y
    std::vector<double> yval{data_y, data_y + size2};
    removeDuplicates(yval);
    // 2gb max heap size
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2147483648);

    cudaMalloc(&devResult, sizeof(double)*yval.size());
    cudaMemset(devResult, 0, sizeof(double)*yval.size());

    std::thread t[threadCount];
    for (int a = 0; a < threadCount; a++) {
        t[a] = std::thread(threadRun, a, k, data_x, size1, data_y, &yval);
    }
    for (int a = 0; a < threadCount; a++) {
        t[a].join();
    }
    double* resultArr = (double*)malloc(sizeof(double) * yval.size());
    cudaMemcpy(resultArr, devResult, sizeof(double)*yval.size(), cudaMemcpyDeviceToHost);
    cudaFree(devResult);

    double result = 0;
    for (int a = 0; a < yval.size(); a++) {
        result += resultArr[a];
    }
    free(resultArr);

    std::cout << "\nFinal Result: " << result << std::endl;
}