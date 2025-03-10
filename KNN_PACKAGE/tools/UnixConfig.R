#!/usr/bin/env Rscript

cat("---- Checking for CUDA/OpenCL/OpenMP support ----\n")

# Use the Unix Makevars file in the src folder.
makevars_file <- file.path("src", "Makevars")

# Remove any existing Makevars file.
if (file.exists(makevars_file)) {
  file.remove(makevars_file)
}

# Prepare local flag variables.
local_cppflags <- character()  # For preprocessor flags: -D... and -I...
local_cxxflags <- character()  # For compiler flags like -fopenmp
local_libs     <- character()  # For libraries: -l..., -L..., etc.
local_ldflags  <- character()  # For additional linker flags if needed

# ------------------------------CHECK FOR CUDA USING NVCC. ------------------

# Define paths for run.cu and run.o within the src folder.
run_cu  <- file.path("src", "cudaKNN.cu")
# compile it directly into src, not into the CUDA folder. makes it easier to find.
run_obj <- file.path("src", "cudaKNN.o")

nvcc <- Sys.which("nvcc")
if (nzchar(nvcc)) {
  message("Found NVCC at: ", nvcc)
  
  # Compile the CUDA file "src/run.cu" to generate "src/run.o".
  # very important that you use this -fPIC option, otherwise it will not make position indepedent code. which you need in a shared library. 
  system2(nvcc, args = c(run_cu,"-arch=sm_50", "-c", "-o", run_obj, "-Xcompiler", "-fPIC"), stdout = TRUE, stderr = TRUE)
  
  if (file.exists(run_obj)) {
    message("Successfully compiled cudaKNN.cu to cudaKNN.o")
    local_libs <- c(local_libs, "cudaKNN.o", "-lcudart")
    local_cppflags <- c(local_cppflags, "-DUSE_CUDA")
  } else {
    warning("Compilation of cudaKNN.cu failed! 'cudaKNN.o' not found.")
  }
} else {
  message("NVCC compiler not found. Continuing without CUDA support.")
}

## --- OpenCL Check ---
# Check for OpenCL support
OPENCL_HOME <- Sys.getenv("OPENCL_HOME")
is_darwin <- Sys.info()["sysname"] == "Darwin" # if we are on mac, it should be supported as a framework.

# For macOS, if OPENCL_HOME is not set, we assume the system's OpenCL framework is available.
if (!nzchar(OPENCL_HOME) && is_darwin) {
  message("OPENCL_HOME not set, but on macOS. Using system OpenCL framework.")
  has_opencl <- TRUE
} else {
  has_opencl <- (nzchar(OPENCL_HOME) || nzchar(Sys.which("clinfo")))
}

if (has_opencl) {
  cat("OpenCL support detected.\n")

  # add our openCL preprocessing flag
  local_cppflags <- c(local_cppflags, "-DUSE_OPENCL")
  # Use OPENCL_HOME if set.
  if (nzchar(OPENCL_HOME)) {
    local_cppflags <- c(local_cppflags, paste0("-I", file.path(OPENCL_HOME, "include")))
    local_libs     <- c(local_libs, paste0("-L", file.path(OPENCL_HOME, "lib")))
  }
  # For Mac use framework linking, for Linux use standard linking.
  if (Sys.info()["sysname"] == "Darwin") {
    local_libs <- c(local_libs, "-framework OpenCL")
  } else {
    local_libs <- c(local_libs, "-lOpenCL")
  }
}

# ------------------------------CHECK FOR OpenMP SUPPORT. ------------------
# Write a temporary test file to try including omp.h.
test_file <- tempfile(fileext = ".c")
writeLines(c("#include <omp.h>", "int main() { return 0; }"), con = test_file)

# Try to compile the test file.
compile_result <- system(paste("R CMD SHLIB", test_file), intern = TRUE)
compiled_so <- sub("\\.c$", .Platform$dynlib.ext, test_file)

if (file.exists(compiled_so)) {
  cat("OpenMP header found, enabling OpenMP flags.\n")
  
  local_cppflags <- c(local_cppflags, "-DHAVE_OPENMP")
  if (Sys.info()["sysname"] == "Darwin") {
    # On macOS with Clang.
    local_cxxflags <- c(local_cxxflags, "-Xpreprocessor", "-fopenmp")
    local_ldflags  <- c(local_ldflags, "-lomp")
  
  } else {
    # On Linux with GCC.
    local_cxxflags <- c(local_cxxflags, "-fopenmp")
    local_ldflags  <- c(local_ldflags, "-fopenmp")
  }
  # removing temp file
  file.remove(compiled_so)

} else {
  cat("omp.h not found, skipping OpenMP.\n")
}
# removing another temp file
file.remove(test_file)

# Get default values from the environment.
default_cpp <- Sys.getenv("SHLIB_CPPFLAGS")
default_cxx <- Sys.getenv("SHLIB_CXXFLAGS")
default_libs <- Sys.getenv("SHLIB_LIBS")

# Build the final flag strings by concatenating defaults with local flags.
final_cpp <- paste(default_cpp, paste(local_cppflags, collapse = " "))
final_cxx <- paste(default_cxx, paste(local_cxxflags, collapse = " "))
final_libs <- paste(default_libs, paste(c(local_ldflags, local_libs), collapse = " "))

# Write the Makevars file with the appropriate flags.
lines <- c(
  paste("PKG_CPPFLAGS =", final_cpp),
  paste("PKG_CXXFLAGS =", final_cxx),
  paste("PKG_LIBS =", final_libs)
)

writeLines(lines, makevars_file)
cat("Makevars file created:\n")
cat(lines, sep = "\n")