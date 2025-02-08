#include <iostream>
#include <cstring>
#include <algorithm>
#include <OpenCL/cl.h>
#include <Rcpp.h>

// Kernel source (placeholder)
const char* kernelSource = "CLC()CLC";

float launchKernel(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y) {
    cl_int err;

    // opencl setup garbage
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    
    size_t sourceSize = std::strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "computeDistance", &err);

    // Get unique y-values and determine group sizes
    Rcpp::NumericVector yVals = Rcpp::unique(data_y);
    int numGroups = yVals.size();
    Rcpp::NumericVector numXsPerY(numGroups);
    
    // Allocate a contiguous host array for all x values (grouped by their y)
    int total_x_size = data_x.size();
    float* x_allValues = new float[total_x_size];
    
    int last_index = 0;
    for (int i = 0; i < numGroups; i++) {
        // Subset data_x corresponding to the current y value
        Rcpp::NumericVector subset_x = data_x[data_y == yVals[i]];
        // Copy the subset for this y value into our array
        std::copy(subset_x.begin(), subset_x.end(), x_allValues + last_index);
        numXsPerY[i] = subset_x.size();
        last_index += subset_x.size();
    }
    
    // Allocate host arrays for distances and results (size as needed)
    float* distances = new float[total_x_size];
    float* results   = new float[total_x_size];
    
    //---Create our OpenCL Buffers ---
    cl_mem xBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * total_x_size, x_allValues, &err);
    
    cl_mem distancesBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * total_x_size, NULL, &err);
    
    cl_mem resultsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * total_x_size, NULL, &err);
    
    // For numXsPerY, we copy data from the Rcpp vector.
    // Note: Rcpp::NumericVector elements are doubles, so convert to int if needed.
    std::vector<int> numXs(numGroups);
    for (int i = 0; i < numGroups; i++) {
        numXs[i] = static_cast<int>(numXsPerY[i]);
    }
    cl_mem numXsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numGroups,numXs.data(), &err);
    
    // --- Set Kernel Arguments ---
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &distancesBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultsBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &numXsBuffer);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &total_x_size);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &numGroups);
    
    // launch our kernel. 
    size_t globalWorkSize = total_x_size;  // For example, one work-item per x value
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(queue);
    
    // get our results back. we actually only need one single float back, since we aggregate the entire results array into one float.
    err = clEnqueueReadBuffer(queue, resultsBuffer, CL_TRUE, 0, sizeof(float) * total_x_size, results, 0, NULL, NULL);
    
    // Cleanup OpenCL objects
    clReleaseMemObject(xBuffer);
    clReleaseMemObject(distancesBuffer);
    clReleaseMemObject(resultsBuffer);
    clReleaseMemObject(numXsBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    // Cleanup host memory
    delete [] x_allValues;
    delete [] distances;
    delete [] results;
    
    return 0;
}
