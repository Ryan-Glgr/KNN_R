options(repos = c(CRAN = "https://cran.rstudio.com/"))
install.packages("Rcpp", repos = "https://cran.rstudio.com/")

# Load required libraries
library("Rcpp")
library("data.table")

Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
Sys.setenv("PKG_LIBS" = "-fopenmp")

# Source C++ code

#sourceCpp("Main_kNN_Implementation.cpp")
sourceCpp("kNN_Test_2.cpp")
#sourceCpp("kNN_Test_3.cpp")
#sourceCpp("kNN_Test_4.cpp")





# Read data using fread for faster processing
Data <- fread("data.csv", skip = 2)

# Extract MATHEFF and MATINTFC from Data
MATHEFF <- Data$MATHEFF
MATINTFC <- Data$MATINTFC

# Define the range of k values
k_values <- seq(5000, 20000, 2500)

# Function to compute IE values
compute_IE <- function(data_x, data_y, ks) {
  sapply(ks, function(k) IE_xy(data_x, data_y, k))
}

# Measure time using Sys.time() for proper elapsed time
start_time <- Sys.time()
result <- compute_IE(MATHEFF[1:50000], MATINTFC[1:50000], k_values)
end_time <- Sys.time()

# Compute correct elapsed time
elapsed_time <- end_time - start_time

# Plot the results
plot(k_values, result, xlab = "k", ylab = "IE(MATHEFF|MATINTFC)")

# Print execution times
cat("Total Execution Time (seconds):\n")
cat("Elapsed time:", elapsed_time, "\n")  # Correct elapsed time

