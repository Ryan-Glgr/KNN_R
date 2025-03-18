/**
 * @file: library.cpp
 * @brief: Creates a connection between each implementation of the problem, and links it to a single "KNN" wrapper function
 *
 * @author: Ryan Gallagher
 * @date: 18.3.202
 *
 */

#include "library.h"
// KNN function is what is available through R, this takes two vectors and a K.
// allows users to choose float or double precision, and allow them to choose an implementation.

/*
 * @param data_x Numeric vector representing the first data set.
 * @param data_y Numeric vector representing the second data set.
 * @param K Integer specifying neighbor in sorted order to use for approximation.
 * @param mode Character string specifying the mode ("cpp", "cuda", or "openCL").
 * @param precision Character string specifying the precision ("float" or "double").
 * @return The result of our Information Energy estimation computation as a double.
 */
// [[Rcpp::export]]
 double KNN(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &mode = "cpp",  const std::string &precision = "float") {
    
    // cuda computation.
    if (mode == "cuda" || mode == "CUDA")
        return cudaKNN(data_x.begin(), data_x.size(), data_y.begin(), data_y.size(), K);

    // openCL implementation. can be run with either floats or double precision. Double precision not supported on some cards, hence the default for floats.
    if (mode == "openCL" || mode == "OpenCL" || mode == "OPENCL" || mode == "opencl")
        return openCL(data_x, data_y, K, precision);

    // pure CPP version
    return IE_xy(data_x, data_y, K);
}