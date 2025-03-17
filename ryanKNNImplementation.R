#!/usr/bin/env Rscript
# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
options(repos = c(CRAN = "https://cran.rstudio.com/"))

if (!requireNamespace("Rcpp", quietly = TRUE)) {
  install.packages("Rcpp")
}
suppressMessages({
  library(Rcpp)
  library(data.table)
  library(CWUKNN)
})

# -------------------------------------------------------------------
# Source the C++ code
# -------------------------------------------------------------------
# 'OpenCL_KNN.cpp' should export a function, e.g., launchKernel(...)
# 'Faster_kNN_Implementation.cpp' should export IE_xy(...)

sourceCpp("Faster_kNN_Implementation.cpp")

# -------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------
Data <- fread("data.csv", skip = 2)
MATHEFF <- Data$MATHEFF
MATINTFC <- Data$MATINTFC

# -------------------------------------------------------------------
# Define CPU and GPU wrappers
# -------------------------------------------------------------------
# CPU approach using IE_xy for a vector of k
compute_IE_Cpp <- function(x, y, ks) {
 sapply(ks, function(k) IE_xy(x, y, k))
}

# Our CPP approach using launchKernel for a vector of k
compute_IE_Ryan_CPP <- function(x, y, ks, mode="", precision="float") {
  sapply(ks, function(k) KNN(x, y, k, mode, precision))
}

# Our CUDA approach using launchKernel for a vector of k
compute_IE_Ryan_CUDA <- function(x, y, ks, mode="cuda", precision="float") {
  sapply(ks, function(k) KNN(x, y, k, mode, precision))
}

# Our OPENCL approach using launchKernel for a vector of k
compute_IE_Ryan_OPENCL <- function(x, y, ks, mode="openCL", precision="float") {
  sapply(ks, function(k) KNN(x, y, k, mode, precision))
}


# -------------------------------------------------------------------
# Define k values and subset data as needed
# -------------------------------------------------------------------
k_values <- c(5000, 10000, 15000, 20000)  # example
N <- 50000  # how many rows you want to test
x_sub <- MATHEFF[1:N]
y_sub <- MATINTFC[1:N]

# -------------------------------------------------------------------
# Timed CPU computation
# -------------------------------------------------------------------
cat("\n--- ANDREW Version Timing ---\n")
t_cpu <- system.time({
 result_cpu <- compute_IE_Cpp(x_sub, y_sub, k_values)
})
cat("ANDREW results:\n")
print(result_cpu)
cat("ANDREW user time:   ", t_cpu["user.self"], " seconds\n")
cat("ANDREW system time: ", t_cpu["sys.self"], " seconds\n")
cat("ANDREW elapsed time:", t_cpu["elapsed"],   " seconds\n\n")

# -------------------------------------------------------------------
# Timed GPU (OpenCL) computation
# -------------------------------------------------------------------
cat("--- CPP Version Timing ---\n")
cwuknn_CPP <- system.time({
  result_gpu_CPP <- compute_IE_Ryan_CPP(x_sub, y_sub, k_values)
})

cat("CPP version results:\n")
print(result_gpu_CPP)
cat("User time:   ", cwuknn_CPP["user.self"], " seconds\n")
cat("System time: ", cwuknn_CPP["sys.self"], " seconds\n")
cat("Elapsed time:", cwuknn_CPP["elapsed"],   " seconds\n\n")

cat("--- CUDA Version Timing ---\n")
cwuknn_CUDA <- system.time({
  result_gpu_CUDA <- compute_IE_Ryan_CUDA(x_sub, y_sub, k_values)
})

cat("CUDA version results:\n")
print(result_gpu_CUDA)
cat("User time:   ", cwuknn_CUDA["user.self"], " seconds\n")
cat("System time: ", cwuknn_CUDA["sys.self"], " seconds\n")
cat("Elapsed time:", cwuknn_CUDA["elapsed"],   " seconds\n\n")

cat("--- OPENCL Version Timing ---\n")
cwuknn_OPENCL <- system.time({
  result_gpu_OPENCL <- compute_IE_Ryan_OPENCL(x_sub, y_sub, k_values)
})

cat("OPENCL version results:\n")
print(result_gpu_OPENCL)
cat("User time:   ", cwuknn_OPENCL["user.self"], " seconds\n")
cat("System time: ", cwuknn_OPENCL["sys.self"], " seconds\n")
cat("Elapsed time:", cwuknn_OPENCL["elapsed"],   " seconds\n\n")

cat("All done.\n")
