#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <CL/cl.h>
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
    // 3) Build Programs and Create Kernels: These will be used to run the OpenCL code.
    //    This is where the file references are made for OpenCL calls.
    //----------------------------------------------------

    // Creates a file input stream called kernelFile which takes in the OpenCL code
    std::ifstream kernelFile("OpenCL_KNN.cl");
    // Error value if there is failure opening the kernel
    if (!kernelFile.is_open()) {
        Rcpp::Rcerr << "Failed to open kernel file \"OpenCL_KNN.cl\"!" << std::endl;
        return -1.0;
    }

    // Reads the kernel file into a string stream
    std::stringstream kernelStream;
    kernelStream << kernelFile.rdbuf();

    // Stores the source code from the file into a c string
    std::string kernelSourceStr = kernelStream.str();
    const char* kernelSource = kernelSourceStr.c_str();

    // Creates and builds the OpenCL program using the read-in source
    size_t sourceSize = std::strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Creates kernels for each function in the OpenCL file
    cl_kernel kernel_dis = clCreateKernel(program, "fillDistanceMatrix", &err);
    cl_kernel kernel_kth = clCreateKernel(program, "kth_element", &err);

    //----------------------------------------------------
    // 4) Unique Groups & Data Setup
    //----------------------------------------------------
    Rcpp::NumericVector yVals = Rcpp::unique(data_y);
    int numGroups = yVals.size(); // Stores the unique number of y values in the dataset

    double globalAccumulator = 0.0;
    int total_x_size = data_x.size();

    // ---------------------------------------------------
    // 5) Iterate Over Each Y-Group
    // ---------------------------------------------------
    for (int g = 0; g < numGroups; g++) {
        // (a) Extract subset of x for yVals[g]
        Rcpp::NumericVector subset_x = data_x[data_y == yVals[g]];
        int groupSize = subset_x.size(); // GroupSize is the size of x that fits within the set of unique ys
        if (groupSize == 0) {
            continue; // skip empty group
        }

        // (b) Copy subset to a float host array
        std::vector<float> hostXGroup(groupSize);

        //
        for (int i = 0; i < groupSize; i++) {
            // cast from double (Rcpp) to float
            hostXGroup[i] = static_cast<float>(subset_x[i]);
        }
        
        // (c) Create device buffers for xGroup & distanceMatrix (float)
        // These will be used for all openCL calls. They store the memory partitions created to run the OpenCL code.
        cl_mem xGroupBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, groupSize * sizeof(float), hostXGroup.data(), &err);
        cl_mem distanceMatrixBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, groupSize * groupSize * sizeof(float), NULL, &err);
        cl_mem resultBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, groupSize * sizeof(double), NULL, &err);
        
        // (d) Calls the OpenCL to fill the distance matrix
        // ------------------------------------------- START DIS ------------------------------------------- //

        // OpenCL_KNN.cl fillDistanceMatrix Call, creates xGroup as input and distanceMatrix as ouput

        // (1) Set kernel args
        err  = clSetKernelArg(kernel_dis, 0, sizeof(cl_mem), &xGroupBuf);
        err |= clSetKernelArg(kernel_dis, 1, sizeof(cl_mem), &distanceMatrixBuf);
        err |= clSetKernelArg(kernel_dis, 2, sizeof(int), &groupSize);

        // (2) Enqueue kernel (2D NDRange) - Runs the kernel program
        size_t globalWorkSize0[2] = { (size_t)groupSize, (size_t)groupSize };
        err = clEnqueueNDRangeKernel(queue, kernel_dis, 2, NULL, globalWorkSize0, NULL, 0, NULL, NULL);
        // Waits for the kernel to finish before executing any more code
        clFinish(queue);

        // -------------------------------------------- END DIS -------------------------------------------- //

        // (e) Calls the OpenCL to get the K smallest element
        // ------------------------------------------- START KTH ------------------------------------------- //

        // OpenCL_KNN.cl kth_element Call, the same memory from KNN is used

        // (1) Set kernel args
        err |= clSetKernelArg(kernel_kth, 0, sizeof(cl_mem), &distanceMatrixBuf);
        err |= clSetKernelArg(kernel_kth, 1, sizeof(cl_mem), &resultBuf);
        err |= clSetKernelArg(kernel_kth, 2, sizeof(int), &groupSize);
        err |= clSetKernelArg(kernel_kth, 3, sizeof(int), &K);

        // (2) Enqueue kernel (2D NDRange) - Runs the kernel program
        size_t globalWorkSize1[2] = { (size_t)groupSize, (size_t)groupSize };
        err = clEnqueueNDRangeKernel(queue, kernel_kth, 1, NULL, globalWorkSize1, NULL, 0, NULL, NULL);
        // Waits for the kernel to finish before executing any more code
        clFinish(queue);

        // -------------------------------------------- END KTH -------------------------------------------- //

        // How is the reduction working if we aren't
        // trimming the arVal array?
        cl_kernel partial_means = clCreateKernel(program, "partial_means", &err);
        cl_kernel partial_mean_sums = clCreateKernel(program, "sum_partial_means", &err);

        // Launch partial_mean kernel
        clSetKernelArg(partial_means, 0, sizeof(cl_mem), &resultBuf);// array
        clSetKernelArg(partial_means, 1, sizeof(int), &groupSize);  // array size




        // Launch parallel sum reduction kernel


        // Attempt a parallel mean calculation
        int workgroupSize = 256;

        // initial size of resultBuf
        int size = groupSize;

        int num_workgroups = (size + workgroup_size - 1) / workgroup_size;
        cl_mem localData = clCreateBuffer(context, CL_MEM_LOCAL, workgroupSize * sizeof(float), NULL, &err);

        clSetKernelArg(sum_partial_means, 0, sizeof(cl_mem), &resultBuf);  // array
        clSetKernelArg(sum_partial_means, 1, sizeof(cl_mem), &localData);  // local memory buffer
        clSetKernelArg(sum_partial_means, 2, sizeof(int), &size);  // array size
        clSetKernelArg(sum_partial_means, 3, sizeof(int), &workgroupSize);  // workgroup size

        clEnqueueNDRangeKernel(queue, sum_partial_means, 1, NULL, &num_workgroups, &workgroup_size, 0, NULL, NULL);

        while(num_workgroups > 1){
            size = num_workgroups;

            num_workgroups = (size + workgroup_size - 1) / workgroup_size;

            clEnqueueNDRangeKernel(queue, sum_partial_means, 1, NULL, &num_workgroups, &workgroup_size, 0, NULL, NULL);
        }

        // Load value to IE
        float IE;
        clEnqueueReadBuffer(queue, arVal, CL_TRUE, 0, sizeof(float), &IE, 0, NULL, NULL);


        /*
        // (f) Read back the N result vector to host in double
        // Creates a vector to store the results
        std::vector<double> resultDbl(groupSize);
        // Reads the results from the OpenCL program
        err = clEnqueueReadBuffer(queue, resultBuf, CL_TRUE, 0, groupSize * sizeof(double), resultDbl.data(), 0, NULL, NULL);
        // Converts the results to a NumericVector
        Rcpp::NumericVector result(resultDbl.begin(), resultDbl.end());
        // (h) Average for this group
        double IE = Rcpp::mean(result);
        */

        // Weighted by group size
        double weight = static_cast<double>(groupSize) / static_cast<double>(total_x_size);
        IE = static_cast<double>(IE);
        // accumulate
        globalAccumulator += IE * weight;

        // (i) Release buffers
        clReleaseMemObject(xGroupBuf);
        clReleaseMemObject(resultBuf);
        clReleaseMemObject(distanceMatrixBuf);
        clReleaseMemObject(localData);
    }

    // ---------------------------------------------------
    // 6) Cleanup & Return
    // ---------------------------------------------------
    clReleaseKernel(kernel_dis);
    clReleaseKernel(kernel_kth);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return globalAccumulator;
}