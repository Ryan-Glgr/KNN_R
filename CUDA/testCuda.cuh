
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


void run (double* data_x, int size1, double* data_y, int size2, int k) {
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
}





















// THE OLD CODE GRAVEYARD IN CASE I NEED IT






//#include "common.cuh"
//
//
//#define threadsPerBlock 32
//
//struct vec {
//    int size;
//    double* data;
//};
//
//
//
//
//
//
//
//__device__ double mean (double* data, int size) {
//    double res = 0;
//    for (int a = 0; a < size; a++) {
//        res += data[a];
//    }
//    return res/size;
//}
//
//__device__ void insertionSort(vec* arr) {
//    for (int i = 1; i < arr->size; ++i) {
//        int key = arr->data[i];
//        int j = i - 1;
//
//        while (j >= 0 && arr->data[j] > key) {
//            arr->data[j + 1] = arr->data[j];
//            j = j - 1;
//        }
//        arr->data[j + 1] = key;
//    }
//}
//
//
//__device__ void equalityVector (vec* data_x, vec* data_y, double check, vec* out) {
//    int size = 0;
//    for (int a = 0; a < data_x->size; a++) {
//        if (data_y->data[a] == check) {
//            out->data[a] = data_x->data[a];
//            size++;
//        }
//    }
//    double* d = (double*)malloc(sizeof(double) * size);
//    int dInd = 0;
//    for (int a = 0; a < out->size; a++) {
//        if (out->data[a] != 0) {
//            d[dInd] = out->data[a];
//            dInd++;
//        }
//    }
//    free(out->data);
//    out->data = d;
//    out->size = size;
//
////    printf("equality vector:\n");
////    for (int a = 0; a < out->size; a++) {
////        printf("%f ", out->data[a]);
////    }
////    printf("\n");
//}
//
//
//
//__global__ void IE_XY (vec data_x, vec data_y, vec yval, int k, float* result) {
//    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//    if (yval.size < index) {
//        vec x{};
//        x.data = (double*)malloc(sizeof(double) * data_x.size);
//        x.size = data_x.size;
//
//        equalityVector(&data_x, &data_y, yval.data[index], &x);
//
//        int N = 10;
//        if(k > N - 1) {
//            k = N - 1;
//        }
//
//        // density vector
//        vec kNN_result{};
//        kNN_result.size = N;
//        kNN_result.data = (double*)malloc(sizeof(double) * N);
//
//        for(int i = 0; i < N; i++) {
//            // distance vector
////            vec dist{x.size, (double*)malloc(sizeof(double)*x.size)};
////            for (int a = 0; a < dist.size; a++) {
////                dist.data[a] = abs(x.data[index] - x.data[a]);
////            }
//////            insertionSort(&dist);
////            float Ri = dist.data[k];
////            kNN_result.data[i] = k / (N * 2 * Ri);
////            free(dist.data);
//        }
//
//        double res = 0;
//        for (int a = 0; a < kNN_result.size; a++) {
//            res += kNN_result.data[a];
//        }
//        res = res/kNN_result.size;
//
//        float IE = mean(kNN_result.data, kNN_result.size);
//        float weight = (float)x.size / (float)data_x.size;
//        atomicAdd(result, IE*weight);
//        free(kNN_result.data);
//        free(x.data);
//    }
//}
//
//
//
//
//
//
//
//
//#include <vector>
//#include <algorithm>
//#include <set>
//
//// code for the function below was provided by this stack overflow article: https://stackoverflow.com/questions/12200486/how-to-remove-duplicates-from-unsorted-stdvector-while-keeping-the-original-or
//// by a user named Yury.
//template<typename T>
//size_t removeDuplicates(std::vector<T>& vec) {
//    std::set<T> seen;
//
//    auto newEnd = std::remove_if(vec.begin(), vec.end(), [&seen](const T& value) {
//        if (seen.find(value) != std::end(seen))
//            return true;
//
//        seen.insert(value);
//        return false;
//    });
//
//    vec.erase(newEnd, vec.end());
//
//    return vec.size();
//}
//
//
//void run (double* data_x, int size1, double* data_y, int size2, int k) {
//
//    // create a vector with only unique elements of data_y
//    std::vector<double> yval{data_y, data_y + size2};
//    removeDuplicates(yval);
//
//    // initialize our arrays on the GPU
//    double* devDataX;
//    double* devDataY;
//    double* devDataYval;
//    cudaMalloc(&devDataX, sizeof(double)*size1);
//    cudaMalloc(&devDataY, sizeof(double)*size2);
//    cudaMalloc(&devDataYval, sizeof(double)*yval.size());
//
//    cudaMemcpy(devDataX, data_x, sizeof(double)*size1, cudaMemcpyHostToDevice);
//    cudaMemcpy(devDataY, data_y, sizeof(double)*size2, cudaMemcpyHostToDevice);
//    cudaMemcpy(devDataYval, yval.data(), sizeof(double)*yval.size(), cudaMemcpyHostToDevice);
//
//    // store the arrays and their sizes into structs because they're neater to use.
//    vec dx{size1, devDataX};
//    vec dy{size2, devDataY};
//    vec dyval{(int)yval.size(), devDataYval};
//
//
//    //
//    float* result;
//    cudaMalloc(&result, sizeof(float));
//    cudaMemset(result, 0, sizeof(float));
//
//    printf("ysize: %zu\n", yval.size());
//    IE_XY<<<ceil((float)yval.size()/threadsPerBlock), threadsPerBlock>>>(dx, dy, dyval, k, result);
//    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
//    cudaDeviceSynchronize();
//
//    float r = 0;
//    cudaMemcpy(&r, result, sizeof(float), cudaMemcpyDeviceToHost);
//    printf("K: %i\nResult: %f\n", k, r);
//
////    const int size = yval.size();
////    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*size);
////    for (int a = 0; a < size; a++) {
////        cudaStreamCreate(&streams[a]);
////
////        kNN<<<1, 1, 0, streams[a]>>>(&devArray_x[a], k);
////    }
////
////    free(streams);
//
//
//
//
//
//
//
//    cudaFree(devDataX);
//    cudaFree(devDataY);
//    cudaFree(devDataYval);
//    cudaFree(result);
////    cudaFree(vectors);
//}