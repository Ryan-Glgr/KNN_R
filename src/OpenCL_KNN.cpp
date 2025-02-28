#pragma message("OPENCL DOESN'T HAVE FLAG DEFINED!")
#ifdef USE_OPENCL
#pragma message ("OPEN CL HAS USE_OPENCL FLAG DEFINED!\n")

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <Rcpp.h>
#include "library.h"

// if we're on Mac, it's OpenCL/cl.h, otherwise the header is CL/cl.h
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define DEBUG false

// ---------------------------------------------------------------------
// 1) EMBEDDED KERNELS AS STRINGS
// ---------------------------------------------------------------------
// Replace these placeholders with your actual OpenCL kernels:

static const char* KERNEL_FILL_DISTANCE = R"CLC(
__kernel void fillDistanceMatrix(__global const T* xGroup,
                                 __global T* distanceMatrix,
                                 const int groupSize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < groupSize && j < groupSize) {
        distanceMatrix[i * groupSize + j] = fabs(xGroup[i] - xGroup[j]);

    }
}
)CLC";

static const char* KERNEL_KTH_ELEMENT = R"CLC(
__kernel void kth_element(__global T* distanceMatrix,
                          __global T* result,
                                 const int groupSize,
                                 const int K)
{

    // Since distanceMatrix is 1D memory in OpenCL, i and groupSize determines the row starting position

    // Declare local variables
    __local int lowerBound;
    __local int upperBound;
    __local int size;
    __local int l;
    __local int j;
    __local int pivotIndex;
    __local T pivot;
    __local T Ri;

    // Get the current thread number (iteration in the "loop")
    const int i = get_global_id(0);

    // Clamp k to the max group size
    const int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;
    // Adjusts k to the correct position in the array
    const int kAdjusted = kClamped + i * groupSize;

    // ------------------------ START QUICKSELECT ------------------------ //

    // Gets the first and last indexes of the current grou
    lowerBound = groupSize * i;
    upperBound = groupSize * i + groupSize - 1;

    // Iterate until the lowerBound and upperBound are the same
    while ( lowerBound <= upperBound ) {

        // --------------- START PARTITION --------------- //

        size = upperBound - lowerBound + 1;
        pivot = distanceMatrix[upperBound];

        l = lowerBound - 1;
        // Partitioning (Lomuto's Algorithm)
        j = lowerBound;
        while (j < upperBound) {

            // If the current is less than the pivot
            // Swap i and the current element
            // And move i up to the next element
            // (Essentially move an element from the right to the left,
            // And then move to the element which will be swapped next)
            if (distanceMatrix[j] < pivot) {
                l++;
                T temp = distanceMatrix[l];
                distanceMatrix[l] = distanceMatrix[j];
                distanceMatrix[j] = temp;
            }

            j++;
        }

        // Swap the last non-swapped on the left and the pivot
        T temp = distanceMatrix[l+1];
        distanceMatrix[l+1] = distanceMatrix[upperBound];
        distanceMatrix[upperBound] = temp;

        // ---------------- END PARTITION ---------------- //

        // Get the current pivot index
        pivotIndex = l + 1;

        // Change the bounds if the pivot is not equal to k
        if (pivotIndex == kAdjusted) {
            break;
        } else if (pivotIndex > kAdjusted) {
            upperBound = pivotIndex - 1;
        } else {
            lowerBound = pivotIndex + 1;
        }

    }

    // ------------------------- END QUICKSELECT ------------------------- //


    // After QuickSelect, store the found element in Ri
    // Calculate row element
    Ri = distanceMatrix[kAdjusted];
    // Calculate the result to be read back to the program
    result[i] = kClamped / (groupSize * 2.0f * Ri);

}
)CLC";

// ---------------------------------------------------------------------
// 2) TEMPLATED FUNCTION TO LAUNCH THE KERNEL(S)
// ---------------------------------------------------------------------
template <typename T>
double launchKernel(Rcpp::NumericVector &data_x, Rcpp::NumericVector &data_y, int K)
{
    cl_int err;

    // 1) Platform/Device
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 2) Create context & command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // -----------------------------------------------------------------
    // 3) Build Programs and Create Kernels
    //    We now use the embedded strings instead of loading a .cl file.
    // -----------------------------------------------------------------
    const char* allKernels[2] = { KERNEL_FILL_DISTANCE, KERNEL_KTH_ELEMENT};
    size_t kernelLengths[2]   = {
        std::strlen(KERNEL_FILL_DISTANCE),
        std::strlen(KERNEL_KTH_ELEMENT),
    };

    // Create the program from the embedded source code
    cl_program program = clCreateProgramWithSource(context,
                                                   2,
                                                   allKernels,
                                                   kernelLengths,
                                                   &err);

    // Build the program, picking float or double via the preprocessor define
    if (std::is_same<T, double>::value) {
        err = clBuildProgram(program, 1, &device, "-D T=double", NULL, NULL);
    } else {
        err = clBuildProgram(program, 1, &device, "-D T=float", NULL, NULL);
    }

    // Create kernels
    cl_kernel kernel_dis = clCreateKernel(program, "fillDistanceMatrix", &err);
    cl_kernel kernel_kth = clCreateKernel(program, "kth_element", &err);

    // 4) Unique Groups & Data Setup
    Rcpp::NumericVector yVals = Rcpp::unique(data_y);
    int numGroups = yVals.size();
    double globalAccumulator = 0.0;
    int total_x_size = data_x.size();

#if DEBUG
    double dis_timer = 0;
    double kth_timer = 0;
    double avg_timer = 0;
#endif

    // 5) Iterate Over Each Y-Group
    for (int g = 0; g < numGroups; g++) {

        // (a) Extract subset of x for yVals[g]
        Rcpp::NumericVector subset_x = data_x[data_y == yVals[g]];
        int groupSize = subset_x.size();
        if (groupSize == 0) {
            continue; // skip empty group
        }

        // (b) Copy subset to a float/double host array
        std::vector<T> hostXGroup(groupSize);
        for (int i = 0; i < groupSize; i++) {
            hostXGroup[i] = static_cast<T>(subset_x[i]);
        }

        // (c) Create device buffers
        cl_mem xGroupBuf = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          groupSize * sizeof(T),
                                          hostXGroup.data(),
                                          &err);

        cl_mem distanceMatrixBuf = clCreateBuffer(context,
                                                 CL_MEM_READ_WRITE,
                                                 groupSize * groupSize * sizeof(T),
                                                 NULL,
                                                 &err);

        cl_mem resultBuf = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE,
                                          groupSize * sizeof(T),
                                          NULL,
                                          &err);

#if DEBUG
        std::clock_t dis_start = std::clock();
#endif

        // (d) fillDistanceMatrix
        err  = clSetKernelArg(kernel_dis, 0, sizeof(cl_mem), &xGroupBuf);
        err |= clSetKernelArg(kernel_dis, 1, sizeof(cl_mem), &distanceMatrixBuf);
        err |= clSetKernelArg(kernel_dis, 2, sizeof(int), &groupSize);

        size_t globalWorkSize0[2] = { (size_t)groupSize, (size_t)groupSize };
        err = clEnqueueNDRangeKernel(queue,
                                     kernel_dis,
                                     2,
                                     NULL,
                                     globalWorkSize0,
                                     NULL,
                                     0,
                                     NULL,
                                     NULL);

        clFinish(queue);

#if DEBUG
        std::clock_t dis_end = std::clock();
        std::clock_t kth_start = std::clock();
#endif

        // (e) kth_element
        err  = clSetKernelArg(kernel_kth, 0, sizeof(cl_mem), &distanceMatrixBuf);
        err |= clSetKernelArg(kernel_kth, 1, sizeof(cl_mem), &resultBuf);
        err |= clSetKernelArg(kernel_kth, 2, sizeof(int), &groupSize);
        err |= clSetKernelArg(kernel_kth, 3, sizeof(int), &K);

        size_t globalWorkSize1[1] = { (size_t)groupSize };
        err = clEnqueueNDRangeKernel(queue,
                                     kernel_kth,
                                     1,
                                     NULL,
                                     globalWorkSize1,
                                     NULL,
                                     0,
                                     NULL,
                                     NULL);

        clFinish(queue);

#if DEBUG
        std::clock_t kth_end = std::clock();
#endif

        // (f) Read back result
        std::vector<T> resultT(groupSize);
        err = clEnqueueReadBuffer(queue,
                                  resultBuf,
                                  CL_TRUE,
                                  0,
                                  groupSize * sizeof(T),
                                  resultT.data(),
                                  0,
                                  NULL,
                                  NULL);

        Rcpp::NumericVector result(resultT.begin(), resultT.end());

#if DEBUG
        std::clock_t avg_start = std::clock();
#endif

        // (g) Average for this group
        double IE = Rcpp::mean(result);
        double weight = static_cast<double>(groupSize) / static_cast<double>(total_x_size);
        globalAccumulator += IE * weight;

        // (h) Release buffers
        clReleaseMemObject(xGroupBuf);
        clReleaseMemObject(resultBuf);
        clReleaseMemObject(distanceMatrixBuf);

#if DEBUG
        std::clock_t avg_end = std::clock();
        dis_timer += double(dis_end - dis_start) / CLOCKS_PER_SEC;
        kth_timer += double(kth_end - kth_start) / CLOCKS_PER_SEC;
        avg_timer += double(avg_end - avg_start) / CLOCKS_PER_SEC;
#endif

    } // end for (group)

#if DEBUG
    std::cout << "-- K: " << K << " --" << std::endl;
    std::cout << "DIS: " << dis_timer << " seconds." << std::endl;
    std::cout << "KTH: " << kth_timer << " seconds." << std::endl;
    std::cout << "AVG: " << avg_timer << " seconds." << std::endl;
#endif

    // 6) Cleanup
    clReleaseKernel(kernel_dis);
    clReleaseKernel(kernel_kth);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return globalAccumulator;
}

// ---------------------------------------------------------------------
// 3) EXTERNALLY VISIBLE FUNCTION WRAPPER
// ---------------------------------------------------------------------
double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type)
{
    // Calls the templated function using the requested numeric type.
    if (type == "double") {
        return launchKernel<double>(data_x, data_y, K);
    } else {
        return launchKernel<float>(data_x, data_y, K);
    }
}

#endif // USE_OPENCL
