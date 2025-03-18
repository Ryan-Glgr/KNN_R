/**
 * @file: CPP_KNN.cpp
 * @brief: Defines the functionality for the CPP implementation
 *
 * @authors: Aaron Snyder, Noah Rodal
 * @date: 3.18.2025
 *
 */

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef HAVE_OPENMP
#include <omp.h>
#else
#include <future>
#endif

using namespace Rcpp;

// Function Signatures
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);
std::vector<double> kNN(const std::vector<double>& data, int k);
void print_vector(const Rcpp::NumericVector& x);

// Function Definitions
/*
  kNN function, which takes a vector of floats and a k value
  returns a vector of the kNN weights for each element of data.
  Uses std::nth_element for more efficient finding of the k-th smallest distance.
*/
std::vector<double> kNN(const std::vector<double>& data, int k) {

    // Set the value of k to N-1 in the case where k > N-1
    int N = data.size();
    if (k > N - 1) k = N - 1;

    // Blank density vector and storage for distances
    std::vector<double> result(N);
    std::vector<double> dist(N);  

    // Loop over data points
    for (int i = 0; i < N; i++) {
        double x_i = data[i];

        // Compute absolute distances in parallel
        for (int j = 0; j < N; j++) {
            dist[j] = std::abs(x_i - data[j]);
        }

        // Find the k-th smallest distance (nth_element)
        std::nth_element(dist.begin(), dist.begin() + k, dist.end());
        double Ri = dist[k];

        // Compute density approximation
        result[i] = k / (N * 2 * Ri);
    }
    return result;
}

/*
  Uses parallel reduction to incrementally calculate the final result.
*/
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k) {
    // Get unique y values
    Rcpp::NumericVector yval = Rcpp::unique(data_y);
    float result = 0;
    int N = data_x.size();

// if we have OpenMP, we are going to use pragmas to handle the multithreading business.
#ifdef HAVE_OPENMP
    // OpenMP implementation
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < yval.size(); i++) {
            std::vector<double> x;
            for (int j = 0; j < data_x.size(); j++) {
                if (data_y[j] == yval[i]) {
                    x.push_back(data_x[j]);
                }
            }
            int x_size = x.size();
            if (x_size > 1) {
                std::vector<double> kNN_result = kNN(x, k);
                float IE = std::accumulate(kNN_result.begin(), kNN_result.end(), 0.0) / kNN_result.size();
                float weight = static_cast<float>(x_size) / static_cast<float>(N);
                #pragma omp atomic
                result += IE * weight;
            }
        }
    }
#else
    // Async/Futures implementation for cross-platform parallelism without OpenMP. 
    std::vector<std::future<float>> futures;
    for (int i = 0; i < yval.size(); i++) {
        // Capture i by value; data_x and data_y are captured by reference since they're read-only
        // exactly same functionality, but we are using futures instead of OpenMP
        futures.push_back(std::async(std::launch::async, [&, i]() -> float {
            std::vector<double> x;
            for (int j = 0; j < data_x.size(); j++) {
                if (data_y[j] == yval[i]) {
                    x.push_back(data_x[j]);
                }
            }
            int x_size = x.size();
            if (x_size > 1) {
                // get our result vector
                std::vector<double> kNN_result = kNN(x, k);

                // add up the numbers in the result vector
                float IE = std::accumulate(kNN_result.begin(), kNN_result.end(), 0.0) / kNN_result.size();
                
                // weight it
                float weight = static_cast<float>(x_size) / static_cast<float>(N);
                
                // return the result of one particular KNN group.
                return IE * weight;
            }
            return 0.0f;
        }));
    }
    // Collect results from futures
    for (auto &f : futures) {
        // this is where we actually receive the values.
        result += f.get();
    }
#endif

    return result;
}