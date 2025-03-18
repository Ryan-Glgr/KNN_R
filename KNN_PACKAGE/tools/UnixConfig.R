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
run_cu  <- file.path("src", "cudaKNN.cu")
# output shared object MUST go in inst, because R doesn't always take all the files from src over. so it goes in a place it must be copied by the installation process
run_so  <- file.path("inst", "kernels", "cudaKNN.so")
nvcc <- Sys.which("nvcc")
if (nzchar(nvcc)) {
  message("Found NVCC at: ", nvcc)

  # IMPORTANT FLAGS! -shared flag creates a shared library and -fPIC makes it position independent.
  system2(nvcc, args = c(run_cu, "-shared", "-rdc=true", "-arch=sm_50", "-std=c++14", "-o", run_so, "-Xcompiler", "-fPIC", "-lcudart", "-O3"),
          stdout = TRUE, stderr = TRUE)

  if (file.exists(run_so)) {
    message("Successfully compiled cudaKNN.cu to cudaKNN.so")
    # Instead of linking the .so in PKG_LIBS, we simply define the USE_CUDA flag.
    local_cppflags <- c(local_cppflags, "-DUSE_CUDA")
  } else {
    warning("Compilation of cudaKNN.cu failed! 'cudaKNN.so' not found.")
  }
} else {
  message("NVCC compiler not found. Continuing without CUDA support.")
}

## --- OpenCL Check ---
# Check for OpenCL support
OPENCL_HOME <- Sys.getenv("OPENCL_HOME")
is_darwin <- Sys.info()["sysname"] == "Darwin" # if we are on mac, it should be supported as a framework.

# For macOS, if OPENCL_HOME is not set, we assume the system's OpenCL framework is available anyways.
if (!nzchar(OPENCL_HOME) && is_darwin) {
  message("OPENCL_HOME not set, but on macOS. Using system OpenCL framework.")
  has_opencl <- TRUE
} else {
  has_opencl <- (nzchar(OPENCL_HOME))
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

# we really have to use this pragma omp parallel in here so we can test if we have the right runtime libraries, not just the header file.
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

# We'll name our output shared library the same way R would by default:
compiled_so <- sub("\\.c$", .Platform$dynlib.ext, test_file)

# Decide which flags to pass depending on platform.
sysname <- Sys.info()[["sysname"]]
if (sysname == "Darwin") {
  # On macOS with Clang and a separate libomp.
  # Typically you need `-Xpreprocessor -fopenmp` plus `-lomp`.
  omp_compile_flags <- c("-Xpreprocessor", "-fopenmp")
  omp_link_flags    <- c("-lomp")
} else {
  # On Linux with GCC (or possibly on Windows with MinGW).
  # Typically you just need `-fopenmp` for both compile and link.
  omp_compile_flags <- c("-fopenmp")
  omp_link_flags    <- c("-fopenmp")
}

# Build a command that invokes R CMD SHLIB with our flags.
# We'll pass them at the end, so they are appended to the compile/link invocation.
compile_command <- paste(
  "R CMD SHLIB",
  shQuote(test_file),
  paste(omp_compile_flags, collapse = " "),
  paste(omp_link_flags, collapse = " ")
)

message("Trying to compile and link OpenMP test program:")
message(compile_command)
compile_result <- tryCatch(
  {
    # Run the command and capture output
    system(compile_command, intern = TRUE)
  },
  error = function(e) {
    # If system call failed, store the message
    return(conditionMessage(e))
  }
)

if (file.exists(compiled_so)) {
  cat("OpenMP compile+link test succeeded. Enabling OpenMP flags.\n")

  # If successful, set the preprocessor definition.
  local_cppflags <- c(local_cppflags, "-DHAVE_OPENMP")

  if (sysname == "Darwin") {
    # On macOS with Clang:
    local_cxxflags <- c(local_cxxflags, "-Xpreprocessor", "-fopenmp")
    # We'll add the link flags to local_ldflags:
    local_ldflags  <- c(local_ldflags, "-lomp")
    # If needed, you might also need -L/path/to/libomp if libomp isn't in a default location.
    # For example:
    # local_ldflags <- c(local_ldflags, "-L/usr/local/opt/libomp/lib", "-lomp")
  } else {
    # On Linux (GCC)
    local_cxxflags <- c(local_cxxflags, "-fopenmp")
    local_ldflags  <- c(local_ldflags, "-fopenmp")
  }

  # remove the successfully compiled test library
  file.remove(compiled_so)
} else {
  cat("OpenMP compile+link test failed or OpenMP unavailable.\n")
  cat("Output from compile attempt:\n")
  print(compile_result)
}
# Finally, remove the temporary source file
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