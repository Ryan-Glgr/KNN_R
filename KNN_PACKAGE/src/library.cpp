#include "library.h"

// KNN function is what is available through R, this takes two vectors and a K.
// additionally, allow users to choose float or double precision, and allow them to choose an implementation.

// [[Rcpp::export]]
double KNN(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &precision = "float", const std::string &mode = "cpp") {

    // cuda computation.
    if (mode == "cuda" || mode == "CUDA")
        return cudaKNN(data_x.begin(), data_x.size(), data_y.begin(), data_y.size(), K);

    // openCL implementation. can be run with either floats or double precision. Double precision not supported on some cards, hence the default for floats.
    if (mode == "openCL" || mode == "OpenCL" || mode == "OPENCL" || mode == "opencl")
        return openCL(data_x, data_y, K, precision);

    // pure CPP version
    return IE_xy(data_x, data_y, K);
}