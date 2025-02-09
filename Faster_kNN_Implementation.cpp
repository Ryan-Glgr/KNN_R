#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define DEBUG false

#if DEBUG
int debug_count = 0;
#endif

/* Function signatures */
// Core functions
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);
Rcpp::NumericVector kNN(Rcpp::NumericVector data, int k);

// Helper functions
void print_vector(Rcpp::NumericVector x);


/* Function Definitions */

/*
  kNN function, which takes a vector of floats and a k value
  returns a vector of the kNN weights for each element of data.
  Uses std::nth_element for more efficient finding of the k-th smallest distance.
*/


// used on a particular point (x,y)'s distance array. sorts the array, and then puts into result array, k / (2N * kth distance value) 
// why 2N???? 
Rcpp::NumericVector kNN(Rcpp::NumericVector data, int k)
{
    int N = data.size();

    // Set the value of k to N-1 in the case where k > N-1
    if (k > N - 1) {
        k = N - 1;
    }

    // Blank density vector
    Rcpp::NumericVector result(N);

    // Thread-local storage for distances
    std::vector<float> dist(N);

    for (int i = 0; i < N; i++)
    {
        // Compute the absolute distances between data[i] and all other elements
        for (int j = 0; j < N; j++) {
            dist[j] = std::abs(data[i] - data[j]);
        }

        // Find the k-th smallest distance using nth_element (more efficient than sorting)
        // rearranges so that we can access the kth element.
        std::nth_element(dist.begin(), dist.begin() + k, dist.end());
        float Ri = dist[k]; // k-th nearest distance

        // Append the kNN-approximated density for the i-th point to the result
        result[i] = k / (N * 2 * Ri);
    }

    #if DEBUG
        Rprintf("kNN result:       ");
        print_vector(result);
    #endif
        return result;
}

/*
  This is the only function exposed to R (via Rcpp::export).
  Optimized relative to the provided R code by using vectorized unique() and cumulative result storage.
  Uses parallel reduction to incrementally calculate the final result.
*/
// [[Rcpp::export]]
float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k)
{
    // Use vectorization to find the unique subset of conditional y-values
    Rcpp::NumericVector yval = Rcpp::unique(data_y);

    // Store the result in a float, avoiding creating two separate arrays for IE and weight
    float result = 0;

    for (int i = 0; i < yval.size(); i++)
    {
        // Use vector logic to find the subset of x data that corresponds to this unique y-value
        Rcpp::NumericVector x = data_x[data_y == yval[i]];

        // Calculate the conditional information energy using the kNN approximation

        // calls our function and gets back a vector and a float for the average of the result. on a particular vector.

        // result is the average distance between the points 
        Rcpp::NumericVector kNN_result = kNN(x, k);
        float IE = Rcpp::mean(kNN_result);

        // Calculate the weight of the conditioned x vector (conditioned by the unique y-value)
        float weight = (float)x.size() / (float)data_x.size();

        // Multiply the information energy by the weight and add it to the result
        result += IE * weight;

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
