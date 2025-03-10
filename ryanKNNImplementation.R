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

# OpenCL approach using launchKernel for a vector of k
compute_IE_Ryan <- function(x, y, ks, precision="float", mode="cpp") {
  sapply(ks, function(k) KNN(x, y, k, precision, mode))
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
cat("--- OUR Version Timing - Float ---\n")
cwuknn <- system.time({
  result_gpu <- compute_IE_Ryan(x_sub, y_sub, k_values)
})

cat("Our version results:\n")
print(result_gpu)
cat("Our user time:   ", cwuknn["user.self"], " seconds\n")
cat("Our system time: ", cwuknn["sys.self"], " seconds\n")
cat("Our elapsed time:", cwuknn["elapsed"],   " seconds\n\n")

cat("All done.\n")
