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
})

# -------------------------------------------------------------------
# Source the C++ code
# -------------------------------------------------------------------
# 'OpenCL_KNN.cpp' should export a function, e.g., launchKernel(...)
# 'Faster_kNN_Implementation.cpp' should export IE_xy(...)
sourceCpp("OpenCL_KNN.cpp")
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

# OpenCL approach using launchKernel for a vector of k
compute_IE_OpenCL <- function(x, y, ks) {
  sapply(ks, function(k) launchKernel(x, y, k))
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
cat("\n--- CPU/C++ Version Timing ---\n")
t_cpu <- system.time({
  result_cpu <- compute_IE_Cpp(x_sub, y_sub, k_values)
})
cat("CPU results:\n")
print(result_cpu)
cat("CPU user time:   ", t_cpu["user.self"], " seconds\n")
cat("CPU system time: ", t_cpu["sys.self"], " seconds\n")
cat("CPU elapsed time:", t_cpu["elapsed"],   " seconds\n\n")

# -------------------------------------------------------------------
# Timed GPU (OpenCL) computation
# -------------------------------------------------------------------
cat("--- OpenCL/GPU Version Timing ---\n")
t_gpu <- system.time({
  result_gpu <- compute_IE_OpenCL(x_sub, y_sub, k_values)
})
cat("GPU/OpenCL results:\n")
print(result_gpu)
cat("GPU user time:   ", t_gpu["user.self"], " seconds\n")
cat("GPU system time: ", t_gpu["sys.self"], " seconds\n")
cat("GPU elapsed time:", t_gpu["elapsed"],   " seconds\n\n")

cat("All done.\n")

