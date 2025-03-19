Information Energy Kth Nearest Neighbor Estimation - C++/CUDA/OpenCL optimization
==================================================
Last README update: 17 March 2025

-----------
DESCRIPTION
-----------
This is an R package which is intended to provide optimized implementations of the Kth 
Nearest Neighbor Information Energy Estimation. There are three different functions which
are available. All three functions perform the same computation, but depending on the particular
machine and problem size, it is advantageous to use CUDA/OpenCL instead of the default C++ implementation.
The function returns a simple double precision float representing the estimated information energy between the two variables.

----------------------------
SET UP STEPS FOR ALL METHODS
----------------------------
THESE REQUIREMENTS GET YOU THE DEFAULT INSTALLATION ONLY USING C++
Make sure that you have:
- R installed
- Rtools installed. Note: Rtools is included by default on some systems but not all. 
    Install from CRAN. 
- Rcpp Library installed as a package on your machine

To use OpenCL functionality, CUDA, or speed up C++ with OpenMP, you need to have a few things.
- For cuda, NVIDIA toolkit installed with NVCC on PATH. you need to set NVCC in your environment variables on your machine.
- For openCL, you need to set OPENCL_HOME on your path on your machine. This is required if you are on any OS other than MacOS trying to use the functionality for OpenCL
- For openMP, you may need to install openMP, or it may come pre bundled with your compiler.

To use KNN function a user must do this in their R script.
    - library(CWUKNN)
    - call the KNN function.
        - arguments to KNN function are as follows:
            - Column of X values
            - Column of Y values
            - Desired K amount (integer)
            - optionally: mode= <"OpenCL"> <"CUDA"> or <"cpp"> with a default using C++
            - optionally: precision="float" or "double" to determine the amount of decimal precision needed.

-------------
FUNCTIONALITY
-------------
This package is an optimized implementation of KNN algorithm for estimating Information Energy.
It is NOT for classification problems requiring KNN style decisions. Tiny K values, 0, 1, etc may result
in bad outputs ie INF or large values. This is simply a mathematical consequence of the K Nearest Neighbor algorithm.
It is designed to be ran with slightly larger k values.