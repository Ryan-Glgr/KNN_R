#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <OpenCL/cl.h>
#include <Rcpp.h>

// [[Rcpp::export]]
double launchKernel(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K)
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

    //----------------------------------------------------
    // 3) Build Program & Create fillDistanceMatrix Kernel
    //----------------------------------------------------
    std::ifstream kernelFile("OpenCL_KNN.cl");
    if (!kernelFile.is_open()) {
        Rcpp::Rcerr << "Failed to open kernel file!" << std::endl;
        return -1.0;
    }
    std::stringstream kernelStream;
    kernelStream << kernelFile.rdbuf();
    std::string kernelSourceStr = kernelStream.str();
    const char* kernelSource = kernelSourceStr.c_str();

    size_t sourceSize = std::strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "fillDistanceMatrix", &err);

    //----------------------------------------------------
    // 4) Unique Groups & Data Setup
    //----------------------------------------------------
    Rcpp::NumericVector yVals = Rcpp::unique(data_y);
    int numGroups = yVals.size();

    double globalAccumulator = 0.0;
    int total_x_size = data_x.size();

    // ---------------------------------------------------
    // 5) Iterate Over Each Y-Group
    // ---------------------------------------------------
    for (int g = 0; g < numGroups; g++) {
        // (a) Extract subset of x for yVals[g]
        Rcpp::NumericVector subset_x = data_x[data_y == yVals[g]];
        int groupSize = subset_x.size();
        if (groupSize == 0) {
            continue; // skip empty group
        }

        // (b) Copy subset to a float host array
        std::vector<float> hostXGroup(groupSize);
        for (int i = 0; i < groupSize; i++) {
            // cast from double (Rcpp) to float
            hostXGroup[i] = static_cast<float>(subset_x[i]);
        }

        // (c) Create device buffers for xGroup & distanceMatrix (float)
        cl_mem xGroupBuf = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          groupSize * sizeof(float),
                                          hostXGroup.data(),
                                          &err);

        cl_mem distanceMatrixBuf = clCreateBuffer(context,
                                                 CL_MEM_READ_WRITE,
                                                 groupSize * groupSize * sizeof(float),
                                                 NULL,
                                                 &err);

        // (d) Set kernel args
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xGroupBuf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &distanceMatrixBuf);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &groupSize);

        // (e) Enqueue kernel (2D NDRange)
        size_t globalWorkSize[2] = { (size_t)groupSize, (size_t)groupSize };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize,
                                     NULL, 0, NULL, NULL);
        clFinish(queue);

        // (f) Read back the NxN distance matrix to host in float
        std::vector<float> hostDistanceMatrix(groupSize * groupSize);
        err = clEnqueueReadBuffer(queue,
                                  distanceMatrixBuf,
                                  CL_TRUE,
                                  0,
                                  groupSize * groupSize * sizeof(float),
                                  hostDistanceMatrix.data(),
                                  0, NULL, NULL);

        // (g) Perform the KNN logic on the host
        Rcpp::NumericVector result(groupSize);
        for (int row = 0; row < groupSize; row++) {
            // rowPtr is float-based
            float* rowPtr = &hostDistanceMatrix[row * groupSize];

            // Suppose groupSize = N
            int kClamped = (K > groupSize - 1) ? (groupSize - 1) : K;

            // Then use kClamped both for nth_element and for the formula
            std::nth_element(rowPtr, rowPtr + kClamped, rowPtr + groupSize);
            float Ri = rowPtr[kClamped];

            result[row] = kClamped / (groupSize * 2.0f * Ri);
        }

        // (h) Average for this group
        double IE = Rcpp::mean(result);

        // Weighted by group size
        double weight = static_cast<double>(groupSize) / static_cast<double>(total_x_size);

        // accumulate
        globalAccumulator += IE * weight;

        // (i) Release buffers
        clReleaseMemObject(xGroupBuf);
        clReleaseMemObject(distanceMatrixBuf);
    }

    // ---------------------------------------------------
    // 6) Cleanup & Return
    // ---------------------------------------------------
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return globalAccumulator;
}
