#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

//device function to partition array
__device__ int partition(double* dist, int left, int right) {
    double pivot = dist[right];  //use rightmost element as pivot
    int i = left - 1;  //pointer to the smaller element

    //swap elements around pivot
    for (int j = left; j <= right - 1; j++) {
        if (dist[j] <= pivot) {
            i++;
            // Swap dist[i] and dist[j]
            double temp = dist[i];
            dist[i] = dist[j];
            dist[j] = temp;
        }
    }

    //swap dist[i + 1] and dist[right]
    double temp = dist[i + 1];
    dist[i + 1] = dist[right];
    dist[right] = temp;

    return (i + 1);  // Return partition index
}

//device function to get distance to k-th nearest element
__device__ double quickselect(double* dist, int left, int right, int k) {
    if (left == right) {
        return dist[left];
    }

    int pivotIndex = partition(dist, left, right);
    
    if (k == pivotIndex) {
        return dist[k];  //return value of k-th smallest element
    } else if (k < pivotIndex) { //search right side of pivot
        return quickselect(dist, left, pivotIndex - 1, k);
    } else { //search left side of pivot
        return quickselect(dist, pivotIndex + 1, right, k);
    }
}

//computes density of single point for each trhead
__global__ void kNNDensity(const double* data, double* result, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //get index
    
    if (i < N) {
        //get distances
        double dist[N]; //allocate local memory to store distances
        for(int j = 0; j < N; j++)
            dist[j] = fabs(data[i] - data[j]);

        //get distance to k-th nearest neighbor
        double Ri = quickselect(dist, 0, N - 1, k);

        //compute density approximation
        result[i] = static_cast<double>(k) / (N * 2 * Ri);
    }
}

//wrapper function to handle CUDA execution
std::vector<double> getDensity(const std::vector<double>& data, int k) {
    int N = data.size();
    
    //set the value of k to N-1 in the case where k > N-1
    if(k > N - 1)
        k = N - 1;

    //allocate device memory
    double *d_data, /* *d_dist, */ *d_result;
    cudaMalloc(&d_data, N * sizeof(double)); //store input data
    cudaMalloc(&d_result, N * sizeof(double)); //store the result for each data point
    
    //copy input data to device
    cudaMemcpy(d_data, data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    //launch CUDA kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    //calculate each elements density approximation in individual threads
    kNNDensity<<<numBlocks, blockSize>>>(d_data, d_result, N, k);
    cudaDeviceSynchronize(); //wait for all threads to finish execution

    //copy result back to host
    std::vector<double> host_result(N);
    cudaMemcpy(host_result.data(), d_result, N * sizeof(double), cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(d_data);
    cudaFree(d_result);

    return host_result;
}

void printVector(const std::vector<double>& vec) {
    std::cout << "[ ";
    for (const double& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::vector<double> data = {3.0, 3.0, 20.0, 17.0, 7.0};
    printf("input:\n");
    printVector(data);

    std::vector<double> density = getDensity(data, 2);
    printf("density result:\n");
    printVector(density);
}