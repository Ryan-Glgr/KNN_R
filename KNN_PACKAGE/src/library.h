#ifndef LIBRARY_H
#define LIBRARY_H

#include <Rcpp.h>

// CUDA version
#ifdef HAVE_CUDA
    extern double cudaKNN(double* vec1, int size1, double* vec2, int size2, int K);
#else
    inline double cudaKNN(double* vec1, int size1, double* vec2, int size2, int K) {
        Rcpp::stop("CUDA support is not available on this system.");
        return 0; // to satisfy the compiler
    }
#endif

// OpenCL version
#ifdef HAVE_OPENCL
    extern double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type);
#else
    inline double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type) {
        Rcpp::stop("OpenCL support is not available on this machine.");
        return 0;
    }
#endif

// Pure C++ implementation declaration.
extern float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);

#endif
