# Load required libraries
library("Rcpp")
library("data.table")

Data <- fread("data.csv", skip = 2)

test1 <- (Data$MATHEFF)[1:100000]
test2 <- (Data$MATINTFC)[1:100000]

size <- length(test2)
k_values <- seq(5000, 20000, 2500)


# load in the CUDA Library
pathToDll <- "CUDA\\cmake-build-debug\\CudaRLibrary.dll"

dyn.load(pathToDll)

rv<-.C("runKernel",
       as.double(unlist(test1)),
       as.integer(size),
       as.double(unlist(test2)),
       as.integer(size),
       as.integer(5000))


dyn.unload(pathToDll)




library("CWUKNN")
system.time({
  print("CUDA")
  resCuda <- KNN(test1, test2, 5000, "double", "cuda")
  print(resCuda)
})
system.time({
  print("CPP")
  resCpp <- KNN(test1, test2, 5000, "float", "cpp")
  print(resCpp)
})

system.time({
  print("OpenCL")
  resCl <- KNN(test1, test2, 5000, "float", "opencl")
  print(resCl)
})
#.libPaths()














#Source C++ code
sourceCpp("CUDA\\original.cpp")

# Read data using fread for faster processing
Data <- fread("data.csv", skip = 2)




# Use Rcpp::export function to perform computation on different k values
compute_IE <- function(data_x, data_y, ks) {
  # Precompute results for each k in the vector ks
  results <- sapply(ks, function(k) IE_xy(data_x, data_y, k))
  return(results)
}

# Extract MATHEFF and MATINTFC from Data
MATHEFF <- Data$MATHEFF
MATINTFC <- Data$MATINTFC
length(MATHEFF)

# Define the range of k values
# k_values <- seq(5000, 20000, 2500)
k_values <- seq(5000, 5000, 2500)

# Measure time and compute IE values
system.time({
  result <- compute_IE(MATHEFF[1:100000], MATINTFC[1:100000], k_values)
})

# Plot the results
plot(k_values, result, xlab = "k", ylab = "IE(MATHEFF|MATINTFC)")
