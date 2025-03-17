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

# Define paths for cudaKNN.cu and the output DLL in the inst/kernels folder.
run_cu  <- file.path("src", "cudaKNN.cu")
run_dll <- file.path("inst", "kernels", "cudaKNN.dll")
nvcc <- Sys.which("nvcc")
# see if we have nvcc. if we do then we try and compile CUDA
if (nzchar(nvcc)) {
  message("Found NVCC at: ", nvcc)

  # Compile the CUDA file into a shared library (.dll).
  system2(nvcc, args = c(shQuote(run_cu), "-shared", "-std=c++14", "-o", shQuote(run_dll), "-lcudart", "-O3"),
          stdout = TRUE, stderr = TRUE)

  # if we find the .dll we can use the -DUSE_CUDA flag, which changes our implementation to load the cuda
  if (file.exists(run_dll)) {
    message("Successfully compiled cudaKNN.cu to cudaKNN.dll")
    # if we found or dll we can just set this flag.
    local_cppflags <- c(local_cppflags, "-DUSE_CUDA")
  } else {
    warning("Compilation of cudaKNN.cu failed! 'cudaKNN.dll' not found.")
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
  local_cppflags <- c(local_cppflags, paste0("-I", shQuote(file.path(OPENCL_HOME, "include"))))
  local_libs     <- c(local_libs, paste0("-L", shQuote(file.path(OPENCL_HOME, "lib"))))

  # use standard linking.
  local_libs <- c(local_libs, "-lOpenCL")
}

# ------------------------------CHECK FOR OpenMP SUPPORT. ------------------
test_file <- tempfile(fileext = ".c")

# trying to test compile a file which uses pragmas. this is needed because we have to find the RUNTIME stuff, not just the omp.h
test_code <- c(
  "#include <stdio.h>",
  "#include <omp.h>",
  "",
  "int main(void) {",
  "    int sum = 0;",
  "    // Use a parallel for loop with a reduction to confirm linking.",
  "    #pragma omp parallel for reduction(+:sum)",
  "    for (int i = 0; i < 100; i++) {",
  "        sum += i;",
  "    }",
  "    printf(\"OpenMP test: sum=%d\\n\", sum);",
  "    return 0;",
  "}"
)
writeLines(test_code, test_file)
# compiled name
compiled_so <- sub("\\.c$", .Platform$dynlib.ext, test_file)

# Attempt to compile + link with OpenMP
compile_cmd <- paste("R CMD SHLIB", shQuote(test_file), "-fopenmp")
message("Trying to compile and link OpenMP test program on Windows:")
message(compile_cmd)

res <- tryCatch(
  system(compile_cmd, intern = TRUE),
  error = function(e) conditionMessage(e)
)

# if we have the file, then we can set the openMP flag. if not we just send the message that we don't have it, and carry on.
if (file.exists(compiled_so)) {
  cat("OpenMP compile+link test succeeded. Enabling OpenMP flags.\n")
  local_cppflags <- c(local_cppflags, "-DHAVE_OPENMP")
  local_cxxflags <- c(local_cxxflags, "-fopenmp")
  local_ldflags  <- c(local_ldflags, "-fopenmp")
  file.remove(compiled_so)
} else {
  cat("OpenMP compile+link test failed or OpenMP unavailable.\n")
  cat("Output from compile attempt:\n")
  print(res)
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