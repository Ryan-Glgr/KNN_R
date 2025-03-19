/*
  Authors: Andrew Struthers, Aaron Snyder, Noah Rodal
  Date: March 2025
  Description: 
  This C++ implementation, integrated with R via Rcpp, provides optimized 
  functions for computing k-Nearest Neighbors (kNN) density estimation and 
  conditional information energy (IE). The kNN function leverages OpenMP 
  for parallelization to efficiently compute neighbor distances, while the 
  IE_xy function estimates conditional information energy using a weighted 
  average approach over unique conditional subsets.
  
  This code is designed for high-performance statistical computing, 
  particularly in machine learning and data analysis applications.
*/

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

#define DEBUG false


#if DEBUG
int debug_count = 0;
#endif

using namespace Rcpp;

// Function Signatures
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);
std::vector<double> kNN(const std::vector<double>& data, int k);
void print_vector(const Rcpp::NumericVector& x);
void computeKNN(const std::vector<double>& data, int k, std::vector<double>& results, int start, int end);

// Function Definitions



/*
  kNN function, which takes a vector of floats and a k value
  returns a vector of the kNN weights for each element of data.
  Uses std::nth_element for more efficient finding of the k-th smallest distance.
*/
void computeKNN(const std::vector<double>& data, int k, std::vector<double>& results, int start, int end) {
    int N = data.size();
    
    for (int i = start; i < end; i++) {
        std::vector<double> dist(N);
        double x_i = data[i];

        // Compute distances
        for (int j = 0; j < N; j++) {
            dist[j] = std::abs(x_i - data[j]);
        }

        // Find k-th smallest distance
        std::nth_element(dist.begin(), dist.begin() + k, dist.end());
        double Ri = dist[k];

        // Compute density approximation
        results[i] = k / (N * 2 * Ri);
    }
}

// Main kNN function (multi-threaded)
std::vector<double> kNN(const std::vector<double>& data, int k) {
    int N = data.size();
    if (k > N - 1) k = N - 1;

    std::vector<double> results(N, 0.0);
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = (N + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // Launch worker threads
    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, N);

        if (start < end) {
            threads.emplace_back(computeKNN, std::ref(data), k, std::ref(results), start, end);
        }
    }

    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return results;
}

/*
  This is the only function exposed to R (via Rcpp::export).
  Optimized relative to the provided R code by using vectorized unique() and cumulative result storage.
  Uses parallel reduction to incrementally calculate the final result.
*/
// [[Rcpp::export]]
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k) {

    // Use vectorization to find the unique subset of conditional y-values
    Rcpp::NumericVector yval = Rcpp::unique(data_y);

    // Store the result in a float, avoiding creating two separate arrays for IE and weight
    float result = 0;

    int N = data_x.size();

    //#pragma omp parallel for reduction(+:result) schedule(dynamic)
    for (int i = 0; i < yval.size(); i++) {
        std::vector<double> x;
        for (int j = 0; j < data_x.size(); j++) {
            if (data_y[j] == yval[i]) {
                x.push_back(data_x[j]);
            }
        }
        int x_size = x.size();  // Store size to avoid recomputation

        if (x_size > 1) {
            std::vector<double> kNN_result = kNN(x, k);
            float IE = std::accumulate(kNN_result.begin(), kNN_result.end(), 0.0) / kNN_result.size();
            float weight = static_cast<float>(x_size) / static_cast<float>(N);
            result += IE * weight;
        }
        #if DEBUG
        Rprintf("Condition vector: ");
        print_vector(x);

        Rprintf("IE[%d]     = %2.6f\n", i, IE);
        Rprintf("weight[%d] = %2.6f\n", i, weight);
        Rprintf("result     = %f\n", result);

        debug_count++;
        if (debug_count > 10)
        {
            break;
        }
#endif
    }

    return result;
}

// Helper function to print vectors, used for debugging
void print_vector(Rcpp::NumericVector x)
{
    Rprintf("{");
    for (int i = 0; i < x.size() - 1; i++)
    {
        Rprintf("%2.4f ", x[i]);
    }
    Rprintf("%2.4f}\n", x[x.size() - 1]);
}