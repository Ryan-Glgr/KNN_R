#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <OpenCL/cl.h>
#include <Rcpp.h>

// [[Rcpp::export]]
float launchKernel(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K) {
    cl_int err;

    // ---------------------- PLATFORM/DEVICE SETUP ----------------------
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Create a context for the device.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    // Create a command queue. (Note: clCreateCommandQueue is deprecated in OpenCL 2.0.)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    

    // Open the kernel file.
    std::ifstream kernelFile("OpenCL_KNN.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file!" << std::endl;
        exit(1); // or handle the error appropriately
    }

    // Read the file into a string.
    std::stringstream kernelStream;
    kernelStream << kernelFile.rdbuf();
    std::string kernelSourceStr = kernelStream.str();
    const char* kernelSource = kernelSourceStr.c_str();

    // ---------------------- PROGRAM BUILDING ----------------------
    size_t sourceSize = std::strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // (For debugging, you may wish to check and print the build log on error.)
    
    cl_kernel kernel = clCreateKernel(program, "computeDistance", &err);

    // ---------------------- DATA PRE-PROCESSING ----------------------
    // Get unique y-values and determine group sizes.
    Rcpp::NumericVector yVals = Rcpp::unique(data_y);
    int numGroups = yVals.size();
    int *numXsPerY = new int[numGroups];
    
    // Allocate a contiguous host array for all x values (grouped by their y).
    int total_x_size = data_x.size();
    float* x_allValues = new float[total_x_size];
    
    int last_index = 0;
    for (int i = 0; i < numGroups; i++) {
        // Shoplifted from Andrew's code.
        // Takes all the x's that correspond to a particular y value.
        Rcpp::NumericVector subset_x = data_x[data_y == yVals[i]];
        // Copy the subset for this y value into our contiguous array.
        std::copy(subset_x.begin(), subset_x.end(), x_allValues + last_index);
        numXsPerY[i] = subset_x.size();
        last_index += subset_x.size();
    }
    
    // Allocate host arrays for distances and results (same total size as x_allValues).
    float* distances = new float[total_x_size];
    float* results   = new float[total_x_size];
    
    // ---------------------- OPENCL BUFFER CREATION ----------------------
    // xBuffer is read-only (and we use CL_MEM_COPY_HOST_PTR to initialize it).
    cl_mem xBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * total_x_size, x_allValues, &err);
    
    // distancesBuffer: read-write because we both write to and later read from it.
    cl_mem distancesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(float) * total_x_size, NULL, &err);
    
    // resultsBuffer: read-write since we'll use it for intermediate per-y results.
    cl_mem resultsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(float) * total_x_size, NULL, &err);
    
    // <<BUG FIX>>: Here, the host pointer is incorrectly passed.
    // Instead of using "numXsBuffer" as the host pointer, it must be "numXsPerY".
    cl_mem numXsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(int) * numGroups, numXsPerY, &err);
    
    // Buffer for the final aggregated result.
    cl_mem finalResult = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);

    // Buffer for the mergeTemp.
    cl_mem mergeTempBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * total_x_size, NULL, &err);


    // ---------------------- SETTING KERNEL ARGUMENTS ----------------------
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &distancesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultsBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &numXsBuffer);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &total_x_size);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &numGroups);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &K);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &finalResult);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &mergeTempBuffer);
    
    // ---------------------- KERNEL LAUNCH ----------------------
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, NULL, NULL, 0, NULL, NULL);
    clFinish(queue);
    
    // ---------------------- READ BACK RESULT ----------------------
    float finalResultValue;
    err = clEnqueueReadBuffer(queue, finalResult, CL_TRUE, 0, sizeof(float),
                              &finalResultValue, 0, NULL, NULL);
    
    // ---------------------- CLEANUP OPENCL OBJECTS ----------------------
    clReleaseMemObject(xBuffer);
    clReleaseMemObject(distancesBuffer);
    clReleaseMemObject(resultsBuffer);
    clReleaseMemObject(numXsBuffer);
    clReleaseMemObject(finalResult);
    clReleaseMemObject(mergeTempBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    // ---------------------- CLEANUP HOST MEMORY ----------------------
    delete [] x_allValues;
    delete [] distances;
    delete [] results;
    delete [] numXsPerY;
    
    return finalResultValue;
}
