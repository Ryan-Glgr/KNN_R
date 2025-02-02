cmake_minimum_required(VERSION 3.30)
project(CudaRLibrary CUDA)

set(CMAKE_CUDA_STANDARD 20)


add_library(CudaRLibrary SHARED
        run.cu
        testCuda.cuh)

set_target_properties(CudaRLibrary PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
