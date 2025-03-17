#!/usr/bin/env Rscript

cat("---- Checking for CUDA/OpenCL/OpenMP support (Windows) ----\n")

# Use the Windows Makevars file in the src folder.
makevars_file <- file.path("src", "Makevars.win")

# Remove any existing Makevars.win file.
if (file.exists(makevars_file)) {
  file.remove(makevars_file)
}

# Prepare local flag variables.
local_cppflags <- character()  # For preprocessor flags: -D... and -I...
local_cxxflags <- character()  # For compiler flags like -fopenmp
local_libs     <- character()  # For libraries: -l..., -L..., etc.
local_ldflags  <- character()  # For additional linker flags if needed

# ------------------------------CHECK FOR CUDA USING NVCC. ------------------
# Define paths for cudaKNN.cu and the output object file in the src folder.
run_cu  <- file.path("src", "cudaKNN.cu")
run_obj <- file.path("src", "cudaKNN.obj")  # Windows uses .obj for object files
nvcc <- Sys.which("nvcc")
if (nzchar(nvcc)) {
  message("Windows User: Found NVCC at: ", nvcc)
  
  # Compile the CUDA file.
  system2(nvcc, args = c(shQuote(run_cu), "-c", "-Xcompiler", "-o", shQuote(run_obj)), stdout = TRUE, stderr = TRUE)
  
  if (file.exists(run_obj)) {
    message("Successfully compiled cudaKNN.cu to cudaKNN.obj")
    
    # Optionally check for CUDA_HOME.
    cuda_home <- Sys.getenv("CUDA_HOME")
    if (nzchar(cuda_home)) {

      # Add include and lib paths (1).
      local_cppflags <- c(local_cppflags, paste0("-I", shQuote(normalizePath(file.path(cuda_home, "include"), winslash = "/"))))
      # Add the compiled CUDA object.
      local_libs <- c(local_libs, "cudaKNN.obj")
      # Add include and lib paths (2).
      local_libs <- c(local_libs, paste0("-L", shQuote(normalizePath(file.path(cuda_home, "lib", "x64"), winslash = "/"))))
      local_libs <- c(local_libs, "-lcudart")

    } else {
      message("CUDA_HOME PATH variable not set! Can not link cuda files, so no CUDA support!")
    }
    
  } else {
    warning("Compilation of cudaKNN.cu failed! 'cudaKNN.obj' not found.")
  }
} else {
  message("NVCC compiler not found. Continuing without CUDA support.")
}

## --- OpenCL Check ---
OPENCL_HOME <- Sys.getenv("OPENCL_HOME")
has_opencl <- nzchar(OPENCL_HOME)

if (has_opencl) {
  cat("OpenCL support detected.\n")

  # Add the OpenCL preprocessor flag.
  local_cppflags <- c(local_cppflags, "-DUSE_OPENCL")

  # Add include and lib paths if OPENCL_HOME is set.
  # Add include and lib paths if OPENCL_HOME is set.
  local_cppflags <- c(local_cppflags, paste0("-I", shQuote(normalizePath(file.path(OPENCL_HOME, "include")))))
  local_libs     <- c(local_libs, paste0("-L", shQuote(normalizePath(file.path(OPENCL_HOME, "lib")))))

  # use standard linking.
  local_libs <- c(local_libs, "-lOpenCL")
}

# ------------------------------CHECK FOR OpenMP SUPPORT. ------------------
# make a temporary test file to try including omp.h.
test_file <- tempfile(fileext = ".c")
# if this fails, we are skipping openMP. this is the easiest way to check.
writeLines(c("#include <omp.h>", "int main() { return 0; }"), con = test_file)

# Try to compile the test file.
compile_result <- system(paste("R CMD SHLIB", test_file), intern = TRUE)
compiled_so <- sub("\\.c$", .Platform$dynlib.ext, test_file)

if (file.exists(compiled_so)) {

  cat("OpenMP header found, enabling OpenMP flags.\n")
  local_cppflags <- c(local_cppflags, "-DHAVE_OPENMP")

  # On Windows we use -fopenmp.
  local_cxxflags <- c(local_cxxflags, "-fopenmp")
  local_ldflags  <- c(local_ldflags, "-fopenmp")
  file.remove(compiled_so)
} else {
  cat("omp.h not found, skipping OpenMP.\n")
}
file.remove(test_file)

# Get default values from the environment.
default_cpp <- Sys.getenv("SHLIB_CPPFLAGS")
default_cxx <- Sys.getenv("SHLIB_CXXFLAGS")
default_libs <- Sys.getenv("SHLIB_LIBS")

# Build the final flag strings by concatenating defaults with our local flags.
final_cpp <- paste(default_cpp, paste(local_cppflags, collapse = " "))
final_cxx <- paste(default_cxx, paste(local_cxxflags, collapse = " "))
final_libs <- paste(default_libs, paste(c(local_ldflags, local_libs), collapse = " "))

# Write the Makevars.win file with the appropriate flags.
lines <- c(
  paste("PKG_CPPFLAGS =", final_cpp),
  paste("PKG_CXXFLAGS =", final_cxx),
  paste("PKG_LIBS =", final_libs)
)

writeLines(lines, makevars_file)
cat("Makevars.win file created:\n")
cat(lines, sep = "\n")