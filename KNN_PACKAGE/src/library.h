//
// Created by Ryan Gallagher on 2/27/25.
//
#ifndef LIBRARY_H
#define LIBRARY_H
#include <Rcpp.h>
#pragma once
// CUDA version
#ifdef USE_CUDA
    extern double cudaKNN(double* vec1, int size1, double* vec2, int size2, int K);
#else
inline double cudaKNN(double* vec1, int size1, double* vec2, int size2, int K) {
    Rcpp::stop("CUDA support is not available on this system.");
}
#endif

// OpenCL version
#ifdef USE_OPENCL
extern double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type);
#else
#pragma message("library.h: Compiling WITHOUT USE_OPENCL!")
inline double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type) {
    Rcpp::stop("OpenCL support is not available on this machine.");
}
#endif

// Pure C++ implementation declaration. not conditional since it's the default.
extern float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);

#endif

