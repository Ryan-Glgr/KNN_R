# Load required libraries
library("Rcpp")
library("data.table")

# load in the CUDA Library
pathToDll <- "CUDA\\cmake-build-debug\\CudaRLibrary.dll"

test1 <- c(1, 2, 3, 4)
test2 <- c(4, 3, 2, 1)
size <- 4

dyn.load(pathToDll)
rv<-.C("runKernel", 
       as.integer(unlist(test1)),
       as.integer(unlist(test2)),
       Result = as.integer(unlist(test)),
       as.integer(size))
dyn.unload(pathToDll)
test <- rv[["Result"]]
test


# Source C++ code
# sourceCpp("Faster_kNN_Implementation.cpp")
#
# # Read data using fread for faster processing
# Data <- fread("data.csv", skip = 2)
#
# # Use Rcpp::export function to perform computation on different k values
# compute_IE <- function(data_x, data_y, ks) {
#   # Precompute results for each k in the vector ks
#   results <- sapply(ks, function(k) IE_xy(data_x, data_y, k))
#   return(results)
# }
#
# # Extract MATHEFF and MATINTFC from Data
# MATHEFF <- Data$MATHEFF
# MATINTFC <- Data$MATINTFC
#
# # Define the range of k values
# k_values <- seq(5000, 20000, 2500)
#
# # Measure time and compute IE values
# system.time({
#   result <- compute_IE(MATHEFF[1:50000], MATINTFC[1:50000], k_values)
# })
#
# # Plot the results
# plot(k_values, result, xlab = "k", ylab = "IE(MATHEFF|MATINTFC)")
