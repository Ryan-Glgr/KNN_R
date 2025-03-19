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
Rcpp::NumericVector kNN(const Rcpp::NumericVector& data, int k);
void print_vector(const Rcpp::NumericVector& x);

// Class for ThreadPool
class ThreadPool {
    public:
        ThreadPool(int num_threads) : num_threads_(num_threads) {
            threads_.reserve(num_threads);
            for (int i = 0; i < num_threads; ++i) {
                threads_.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            condition_.wait(lock, [this] { return !tasks_.empty() || should_stop_; });
                            if (should_stop_ && tasks_.empty()) {
                                return;
                            }
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }
    
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                should_stop_ = true;
            }
            condition_.notify_all();
            for (auto& thread : threads_) {
                thread.join();
            }
        }
    
        template <typename F, typename... Args>
        void enqueue(F&& f, Args&&... args) {
            std::function<void()> task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                tasks_.push(std::move(task));
            }
            condition_.notify_one();
        }
    
    private:
    
        std::vector<std::thread> threads_;
        std::queue<std::function<void()>> tasks_;
        std::mutex mutex_;
        std::condition_variable condition_;
        int num_threads_;
        bool should_stop_ = false;
    
};
    
    

// Function Definitions
/*
  kNN function, which takes a vector of floats and a k value
  returns a vector of the kNN weights for each element of data.
  Uses std::nth_element for more efficient finding of the k-th smallest distance.
*/
void computeDistances(int i, const Rcpp::NumericVector& data, std::vector<double>& dist) {
    double x_i = data[i];
    for (int j = 0; j < data.size(); j++) {
        dist[j] = std::abs(x_i - data[j]);
    }
}

Rcpp::NumericVector kNN(const Rcpp::NumericVector& data, int k) {
    int N = data.size();
    if (k > N - 1) k = N - 1;

    Rcpp::NumericVector result(N);
    ThreadPool pool(2);

    for (int i = 0; i < N; i++) {
        std::vector<double> dist(N);
        pool.enqueue(computeDistances, i, data, std::ref(dist));
        
        std::nth_element(dist.begin(), dist.begin() + k, dist.end());
        double Ri = dist[k];
        result[i] = k / (N * 2 * Ri);
    }

    return result;
}

/*
  This is the only function exposed to R (via Rcpp::export).
  Optimized relative to the provided R code by using vectorized unique() and cumulative result storage.
  Uses parallel reduction to incrementally calculate the final result.
*/
std::mutex result_mutex;  
// Thread function to compute partial IE_xy results
void computePartialIE(int start, int end, 
                      const Rcpp::NumericVector& data_x, 
                      const Rcpp::NumericVector& data_y, 
                      const Rcpp::NumericVector& yval, 
                      int k, float& result, int N) {
    float local_result = 0;  // Thread-local result to avoid conflicts

    for (int i = start; i < end; i++) {
        Rcpp::NumericVector x = data_x[data_y == yval[i]];
        int x_size = x.size();

        if (x_size > 1) {
            Rcpp::NumericVector kNN_result = kNN(x, k);
            float IE = Rcpp::mean(kNN_result);
            float weight = static_cast<float>(x_size) / static_cast<float>(N);
            local_result += IE * weight;
        }
    }

    // Safely update global result
    std::lock_guard<std::mutex> lock(result_mutex);
    result += local_result;
}

// [[Rcpp::export]]
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k) {
    Rcpp::NumericVector yval = Rcpp::unique(data_y);
    float result = 0;
    int N = data_x.size();

    int num_threads = std::thread::hardware_concurrency();  
    int num_elements = yval.size();
    int chunk_size = (num_elements + num_threads - 1) / num_threads;  \

    std::vector<std::thread> threads;
    std::vector<float> partial_results(num_threads, 0);

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, num_elements);

        if (start < end) {  
            threads.emplace_back(computePartialIE, start, end, 
                                 std::ref(data_x), std::ref(data_y), std::ref(yval), 
                                 k, std::ref(partial_results[i]), N);
        }
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Sum up partial results
    for (float partial : partial_results) {
        result += partial;
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