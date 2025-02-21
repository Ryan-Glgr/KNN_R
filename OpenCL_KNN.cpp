#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <ctime>
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

    // Calculate time points
    double dis_timer = 0;
    double kth_timer = 0;
    double avg_timer = 0;

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

        
        // Get starting timepoint
        std::clock_t dis_start = std::clock();

        
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

        // Get ending timepoint
        std::clock_t dis_end = std::clock();
        // Get starting timepoint
        std::clock_t kth_start = std::clock();

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

        // Get ending timepoint
        std::clock_t kth_end = std::clock();

        // (f) Read back the N result vector to host in double
        
        // Creates a vector to store the results
        std::vector<double> resultDbl(groupSize);
        // Reads the results from the OpenCL program
        err = clEnqueueReadBuffer(queue, resultBuf, CL_TRUE, 0, groupSize * sizeof(double), resultDbl.data(), 0, NULL, NULL);
        // Converts the results to a NumericVector
        Rcpp::NumericVector result(resultDbl.begin(), resultDbl.end());

        
        // Get starting timepoint
        std::clock_t avg_start = std::clock();
        
        // (h) Average for this group
        double IE = Rcpp::mean(result);

        // Get ending timepoint
        std::clock_t avg_end = std::clock();

        // Weighted by group size
        double weight = static_cast<double>(groupSize) / static_cast<double>(total_x_size);

        // accumulate
        globalAccumulator += IE * weight;

        // (i) Release buffers
        clReleaseMemObject(xGroupBuf);
        clReleaseMemObject(resultBuf);
        clReleaseMemObject(distanceMatrixBuf);

        // Adds time to dis_timer and kth_timer
        dis_timer += double(dis_end - dis_start) / CLOCKS_PER_SEC;
        kth_timer += double(kth_end - kth_start) / CLOCKS_PER_SEC;
        avg_timer += double(avg_end - avg_start) / CLOCKS_PER_SEC;

    }

    std::cout << "-- K: " << K << " --" << std::endl;
    std::cout << "DIS: " << dis_timer << " seconds." << std::endl;
    std::cout << "KTH: " << kth_timer << " seconds." << std::endl;
    std::cout << "AVG: " << avg_timer << " seconds." << std::endl;

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