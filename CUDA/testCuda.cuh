#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <thrust/sort.h>
#include <chrono>

// sets if this program will be ran with naive profiling.
#define PROFILE false

// the amount of threads per block when CUDA kernels are called.
#define threadsPerBlock 128

// how many CPU threads will be ran with their own default CUDA stream.
#define threadCount 1

// heap size
#define heapSize 2147483648

auto profileClock = std::chrono::high_resolution_clock::now(); //! the clock that will help keep track of the runtime.
long double runTime = 0; //! the runtime of this application.

__device__ double* devResult;


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



/*! \fn long double calculateTime ()
 *  \brief calculates the time from the given start time to now, resets the start time to now, then returns as seconds the time elapsed. will change the value of profileClock and runTime as a side effect.
 *  \return time that has elapsed since the last time calculateTime() was called, in microseconds.
 */
long double calculateTime () {
    auto end = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - profileClock);
    profileClock = std::chrono::high_resolution_clock::now();
    long double val = (long double)end.count()/1000000.0;
    runTime += val;
    return val;
}

/*! \fn size_t removeDuplicates(std::vector<T>& vec)
 *  \brief removes the duplicates from a vector.
 *  \author code for the function below was provided by this stack overflow article: https://stackoverflow.com/questions/12200486/how-to-remove-duplicates-from-unsorted-stdvector-while-keeping-the-original-or by a user named Yury.
 *  \param vec an std::vector. this vector will be changed as a side effect of the function.
 *  \return the size of the new vector.
 */
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

/*! \fn __global__ void distanceVectorize (double* x, const int size)
 *  \brief takes a matrix of doubles as a pointer, then based on index it will traverse its length, finding the distance to the index value.
 *  \param x a matrix of doubles as a pointer. will be changed as a side effect.
 *  \param size the size of a slice of the matrix.
 *  \return void, but will change x as a side effect.
 */
__global__ void distanceVectorize (double* x, const int size, const int offset) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i+offset < size) {
        // distance vector
        int pos = i * size;
        double tempVal = x[pos+i+offset];
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
__global__ void gpukNN (double* x, const int size, const int k, double* kernelResult, const double weight, const int ind) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    // block memory for reduction
    __shared__ double* blockResult;
    if (threadIdx.x == 0) {
        blockResult = (double*)malloc(sizeof(double));
        *blockResult = 0;
    }

    // knn calculation
    if (i < size && ind+i < size) {
        int pos = i * size;
        //sort
        thrust::sort(thrust::cuda::par.on(0), &x[pos], &x[pos + size]);
        //
        atomicDoubleAdd(blockResult, (k / (size * 2 * x[pos + k])));
    }
    __syncthreads();

    // reduce
    if (threadIdx.x == 0) {
        atomicDoubleAdd(kernelResult, *blockResult / size * weight);
        free(blockResult);
    }
}

/*! \fn __global__ void multiWrite (double* memoryLocation, const double* values, const int size).
 *  \brief writes a vector into a memory location a number of times in parralel, dependant on the kernel launch size.
 *  \param memoryLocation the space the vector is written into, as a double pointer.
 *  \param values the vector to write into memory location.
 *  \param size the size of the vector.
 *  \return void, but memoryLocation will be changed as a side effect.
 */
__global__ void multiWrite (double* memoryLocation, const double* values, const int size, const int offset) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i+offset < size) {
        memcpy(&memoryLocation[i*size], values, sizeof(double)*size);
    }
}

/*! \fn void kNN (const std::vector<double> x, int k, const int writeLocation, const double weight, double* devMatrixX, int* prevMatrixXSize)
 *  \brief this function sets up the GPU to perform the KNN calculation, then launches the multiWrite, distanceVectorize and gpuKNN kernels to do that calculation.
 *  \param x an equality vector, calculated inside of threadRun().
 *  \param k the kth nearest neighbor to grab.
 *  \param weight the weight used to determine the influence of the calculated Information Energy.
 *  \return void
 */
void kNN (const std::vector<double> x, int k, const int writeLocation, const double weight) {
    int N = x.size();

    if(k > N - 1) {
        k = N - 1;
    }

    if (N <= 1) {
        return;
    }

    // To future maintainers: to reduce the burden of the heap to allocate devMatrixX on obscenely sized datasets,
    // try slicing up the devMatrix in this function, and running in separate passes.

    #if PROFILE
        std::cout << std::fixed <<"\tKNN Pre-Malloc: " << calculateTime() << std::endl;
        std::cout << "\t\tMatrix Memory: " << N*N*sizeof(double) << std::endl;
    #endif



    // vector to copy acrossed devMatrixX
    double* devX;
    cudaMalloc(&devX, sizeof(double) * N);
    cudaMemcpy(devX, x.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    // how many passes we will be doing as to not blow up our heap if our N is large
    int amtOfChunks = (int)ceil(((float)N*N*2)*sizeof(double) / heapSize);

    // where each iteration of the loop will start
    int offset = 0;

    // how many blocks will we be using for each iteration
    int blocks = ceil(((float)N/amtOfChunks)/threadsPerBlock);

    // create a matrix to hold our vectors on the GPU
    double* devMatrixX;
    cudaMalloc(&devMatrixX, sizeof(double) * blocks * threadsPerBlock * N);

    for (int a = 0; a < amtOfChunks; a++) {
        // write our vectors into the matrix
        multiWrite<<<blocks, threadsPerBlock>>>(devMatrixX,devX,N,offset);

        // do the distance calculation on these vectors
        distanceVectorize<<<blocks, threadsPerBlock>>>(devMatrixX,N,offset);

        // sort and get the kth value
        gpukNN<<<blocks, threadsPerBlock>>>(devMatrixX,N,k,&devResult[writeLocation],weight,offset);

        // update our offset and continue
        offset += blocks * threadsPerBlock;
    }



    cudaFree(devX);
    cudaFree(devMatrixX);
    #if PROFILE
        std::cout << std::fixed << "\tKNN end: " << calculateTime() << std::endl;
    #endif
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
void threadRun (const int index, int k, const double* data_x, int size1, const double* data_y, const std::vector<double>* yval) {
    // iterate through yval, starting at a given index and skipping indices based on threadCount.
    for (int i = index; i < yval->size(); i += threadCount) {
        #if PROFILE
                std::cout << std::fixed << "Iteration " << i << " START:" << calculateTime() << std::endl;
        #endif

        //equality vector calculation
        std::vector<double> x{};
        for (int a = 0; a < size1; a++) {
            if (data_y[a] == yval->at(i)) {
                x.push_back(data_x[a]);
            }
        }

        // calculate the information energy
        kNN(x, k, i, (double) x.size() / (double) size1);
        #if PROFILE
                std::cout << std::fixed << "Iteration " << i << " END:" << calculateTime() << std::endl;
        #endif
    }
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
void run (double* data_x, int size1, double* data_y, int size2, int k) {
    // create the yval vector where yval is all singleton values of data_y
    std::vector<double> yval{data_y, data_y + size2};
    removeDuplicates(yval);
    calculateTime();

    // 2gb max heap size
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);

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

    double result = 0;
    for (int a = 0; a < yval.size(); a++) {
        result += resultArr[a];
    }
    free(resultArr);

    std::cout << "\nFinal Result: " << result << std::endl;
}