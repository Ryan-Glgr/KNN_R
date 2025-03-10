#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <thrust/sort.h>
#include <thread>

// sets if this program will be ran with naive profiling.
#define PROFILE false

// the amount of threads per block when CUDA kernels are called.
#define threadsPerBlock 128

// how many CPU threads will be ran with their own default CUDA stream.
#define threadCount 1

double* devResult;

/*! \fn __device__ double atomicDoubleAdd(double* address, double val).
 *  \brief does atomic addition on the double data type, as some cards do not support this inside of CUDA at the time of writing. function definition taken from the Nvidia documentation.
 *  \author Nvidia
 *  \param address the address of the double val is added to.
 *  \param val the value that will be added to the double at the given address.
 *  \return the old value stored at the address.
 */
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

/*! \fn size_t removeDuplicates(std::vector<T>& vec)
 *  \brief removes the duplicates from a vector.
 *  \author code for the function below was provided by this stack overflow article: https://stackoverflow.com/questions/12200486/how-to-remove-duplicates-from-unsorted-stdvector-while-keeping-the-original-or by a user named Yury.
 *  \param vec an std::vector. this vector will be changed as a side effect of the function.
 *  \return the size of the new vector.
 */
 template<typename T>
__host__ size_t removeDuplicates(std::vector<T>& vec) {
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

/*! \fn __global__ void distanceVectorize (double* x, const int size)
 *  \brief takes a matrix of doubles as a pointer, then based on index it will traverse its length, finding the distance to the index value.
 *  \param x a matrix of doubles as a pointer. will be changed as a side effect.
 *  \param size the size of a slice of the matrix.
 *  \return void, but will change x as a side effect.
 */
__global__ void distanceVectorize (double* x, const int size) {
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

/*! \fn __global__ void gpukNN (double* x, const int size, const int k, double* kernelResult, const double weight)
 *  \brief calculates the kth nearest neighbor of a given array inside of a flattened matrix.
 *  \param x a flattened matrix of doubles, as a pointer.
 *  \param size size of an array inside the matrix x.
 *  \param k the kth value to grab.
 *  \param kernelResult the global memory location to write the result into.
 *  \param weight the weight used to determine the influence of the calculated Information Energy.
 *  \return void
 */
__global__ void gpukNN (double* x, const int size, const int k, double* kernelResult, const double weight) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    // block memory for reduction
    extern __shared__ double blockResult[];
    
    if (threadIdx.x == 0) {
        blockResult[0] = 0;
    }
    // knn calculation
    if (i < size) {
        int pos = i * size;
        //sort
        thrust::sort(thrust::seq, &x[pos], &x[pos+size]);
        //
        atomicDoubleAdd(&blockResult[0], (k / (size * 2 * x[pos+k])));
    }
    __syncthreads();

    // reduce
    if (threadIdx.x == 0) {
        atomicDoubleAdd(kernelResult, blockResult[0] / size * weight);
    }
}   

/*! \fn __global__ void multiWrite (double* memoryLocation, const double* values, const int size).
 *  \brief writes a vector into a memory location a number of times in parralel, dependant on the kernel launch size.
 *  \param memoryLocation the space the vector is written into, as a double pointer.
 *  \param values the vector to write into memory location.
 *  \param size the size of the vector.
 *  \return void, but memoryLocation will be changed as a side effect.
 */
__global__ void multiWrite (double* memoryLocation, const double* values, const int size) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        memcpy(&memoryLocation[i*size], values, sizeof(double)*size);
    }
}

/*! \fn void kNN (const std::vector<double> x, int k, const int writeLocation, const double weight, double* devMatrixX, int* prevMatrixXSize)
 *  \brief this function sets up the GPU to perform the KNN calculation, then launches the multiWrite, distanceVectorize and gpuKNN kernels to do that calculation.
 *  \param x an equality vector, calculated inside of threadRun().
 *  \param k the kth nearest neighbor to grab.
 *  \param weight the weight used to determine the influence of the calculated Information Energy.
 *  \param devMatrixX a flattened array that is used to store x.size() amount of equality vectors. Potentially reallocated as a side effect to fit these vectors during runtime.
 *  \param prevMatrixXSize denotes how many values devMatrixX can hold, as a pointer due to potentially needing to be changed as a side effect during runtime.
 *  \return void
 */
__host__ void kNN (const std::vector<double> x, int k, const int writeLocation, const double weight, double** devMatrixX, int* prevMatrixXSize) {
    
    int N = x.size();
    if(k > N - 1) {
        k = N - 1;
    }

    if (N <= 1) {
        return;
    }

    // To future maintainers: to reduce the burden of the heap to allocate devMatrixX on obscenely sized datasets,
    // try slicing up the devMatrix in this function, and running in separate passes.
    if (*prevMatrixXSize < N*N) {
        cudaFree(*devMatrixX);
        cudaMalloc(devMatrixX, sizeof(double) * N * N);
        *prevMatrixXSize = N * N;  // update the recorded size
    }

    // vector to copy acrossed devMatrixX
    double* devX;
    cudaMalloc(&devX, sizeof(double) * N);
    cudaMemcpy(devX, x.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    // have the kernel threads call memcpy N times
    printf("\n\nLAUNCHING KERNELS WITH N = %d\t NUMBLOCKS = %d\t BLOCKSIZE = %d\n", N, (int)ceil((float)N/threadsPerBlock), threadsPerBlock);
    fflush(stdout);
    multiWrite<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(*devMatrixX, devX, N);
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    //// do kNN calculation
    // calculate the distance vectors in parallel
    distanceVectorize<<<ceil((float)N/threadsPerBlock), threadsPerBlock>>>(*devMatrixX, N);
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // sort the array then grab the kth element. these values are then reduced and written
    // to a spot in devResult.
    gpukNN<<<ceil((float)N/threadsPerBlock), threadsPerBlock, sizeof(double)>>>(*devMatrixX, N, k, &devResult[writeLocation], weight);
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaFree(devX);
}

/*!
 *  \fn void threadRun (int index, int k, double* data_x, int size1, double* data_y, std::vector<double>* yval)
 *  \brief sets up and runs kNN on a thread.
 *  \param index the thread index, for when this application is ran multi-threaded.
 *  \param k the kth nearest neighbor to grab.
 *  \param data_x the first column of data.
 *  \param size1 the number of values inside of the first column of data.
 *  \param data_y the second column of data. data_y should *at least* be as long as data_x.
 *  \param yval a vector of unique values found inside of data_y.
 *  \return void
 */
__host__ void threadRun (const int index, int k, const double* data_x, int size1, const double* data_y, const std::vector<double>* yval) {
    // allocate a matrix that will have our x vector copied into it many times.
    // kNN will change this matrices size as a side-effect if it needs to be bigger during runtime.
    double* devMatrixX;
    int prevMatrixXSize = yval->size()*yval->size();
    cudaMalloc(&devMatrixX, sizeof(double)*prevMatrixXSize);

    // iterate through yval, starting at a given index and skipping indices based on threadCount.
    for (int i = index; i < yval->size(); i += threadCount) {

        //equality vector calculation
        std::vector<double> x{};
        for (int a = 0; a < size1; a++) {
            if (data_y[a] == yval->at(i)) {
                x.push_back(data_x[a]);
            }
        }
        // calculate the information energy
        kNN(x, k, i, (double) x.size() / (double) size1, &devMatrixX, &prevMatrixXSize);
    }
    cudaFree(devMatrixX);
}

/*!
 * \fn void run (double* data_x, int size1, double* data_y, int size2, int k)
 * \brief function ran by runKernel that sets up and runs a number of threads defined by threadCount to run threadRun, then sums and prints out the answer.
 * \param data_x the first column of data.
 * \param size1 the amount of values inside of the first column of data.
 * \param data_y the second column of data.
 * \param size2 the amount of values inside of the second column of data.
 * \param k the kth nearest neighbor to find.
 */
__host__ double run (double* data_x, int size1, double* data_y, int size2, int k) {
    // create the yval vector where yval is all singleton values of data_y
    std::vector<double> yval{data_y, data_y + size2};
    removeDuplicates(yval);

    cudaMalloc(&devResult, sizeof(double)*yval.size());
    cudaMemset(devResult, 0, sizeof(double)*yval.size());

    // create and launch N threads, they'll each run a version of threadRun.
    // if there are 3 threads, thread 1->3 will start on 1->3 respectively,
    // then increment by 3 each iteration inside of threadRun().
    std::thread t[threadCount];
    for (int a = 0; a < threadCount; a++) {
        t[a] = std::thread(threadRun, a, k, data_x, size1, data_y, &yval);
    }
    for (int a = 0; a < threadCount; a++) {
        t[a].join();
    }

    // create the result array that will be summed up. each run of kNN will write into this buffer
    // to avoid having to memcpy each iteration, so it is retrieved here.
    double* resultArr = (double*)malloc(sizeof(double) * yval.size());
    cudaMemcpy(resultArr, devResult, sizeof(double)*yval.size(), cudaMemcpyDeviceToHost);
    cudaFree(devResult);

    double result = 0.0;
    for (int a = 0; a < yval.size(); a++) {
        result += resultArr[a];
    }
    free(resultArr);

    return result;
}