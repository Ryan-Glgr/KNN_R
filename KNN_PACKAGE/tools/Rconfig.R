#!/usr/bin/env Rscript

cat("---- Checking for CUDA/OpenCL/OpenMP support ----\n")

# Determine which Makevars file extension to write for Windows or Unix
makevars_file <- if (.Platform$OS.type == "windows") {
	file.path("src", "Makevars.win")
} else {
	file.path("src", "Makevars")
}

# Remove any existing Makevars file so we can remake it.
if (file.exists(makevars_file)) {
	file.remove(makevars_file)
}

# Prepare local flag variables:
local_cppflags <- character()	# For preprocessor flags: -D... and -I...
local_cxxflags <- character()	# For compiler flags like -fopenmp or /openmp
local_libs     <- character()	# For libraries: -l..., -L..., etc.
local_ldflags  <- character()	# For additional linker flags if needed

# ------------------------------CHECK FOR CUDA USING NVIDIA-SMI. ------------------
nvidia_smi <- Sys.which("nvidia-smi")
if (nvidia_smi != "") {
	# run the nvidia-smi command to see if we have an nvidia card.
	out <- system2(nvidia_smi, args = "--query-gpu=name --format=csv,noheader", stdout = TRUE, stderr = TRUE)
	# if we get any output from that.
	if (length(out) > 0 && any(nzchar(out))) {
		# set USE_CUDA flag for compilation time.
		message("Detected NVIDIA GPU(s):")
		local_cppflags <- c(local_cppflags, "-DUSE_CUDA")
		# On Windows, use Windows-style pathing for the includes.
		if (.Platform$OS.type == "windows") {
			# Check for CUDA_PATH
			CUDA_PATH <- Sys.getenv("CUDA_PATH")
			if (nzchar(CUDA_PATH)) {
				message("CUDA_PATH detected: ", CUDA_PATH)
				local_cppflags <- c(local_cppflags, paste0("-I'", file.path(CUDA_PATH, "include"), "'"))
				# Windows compilers often prefer /LIBPATH: flags.
				local_libs <- c(local_libs, paste0("/LIBPATH:'", file.path(CUDA_PATH, "lib", "x64"), "'"), "cudart", "cuda")
			} else {
				warning("CUDA detected but CUDA_PATH not set on Windows!")
			}
		} else {
			# Unix/Linux default paths
			CUDA_HOME <- Sys.getenv("CUDA_HOME")
			if (nzchar(CUDA_HOME)) {
				message("CUDA_HOME detected: ", CUDA_HOME)
				local_cppflags <- c(local_cppflags, paste0("-I", CUDA_HOME, "/include"))
			} else if (file.exists("/usr/local/cuda/include/cuda.h")) {
				message("Using default CUDA include path: /usr/local/cuda/include")
				local_cppflags <- c(local_cppflags, "-I/usr/local/cuda/include")
			} else {
				warning("CUDA detected but CUDA include directory not found!")
			}
			# Link the CUDA libraries for the compiler.
			if (nzchar(CUDA_HOME) && dir.exists(file.path(CUDA_HOME, "lib64"))) {
				local_libs <- c(local_libs, paste0("-L", file.path(CUDA_HOME, "lib64")), "-lcudart", "-lcuda")
			} else if (file.exists("/usr/local/cuda/lib64")) {
				local_libs <- c(local_libs, "-L/usr/local/cuda/lib64", "-lcudart", "-lcuda")
			}
		}
	} else {
		message("nvidia-smi ran but didn't report any GPUs.")
	}
} else {
	message("nvidia-smi not available. continuing without CUDA functionality.")
}

## --- OpenCL Check ---
OPENCL_HOME <- Sys.getenv("OPENCL_HOME")
has_opencl <- (nzchar(OPENCL_HOME) || nzchar(Sys.which("clinfo")))
if (has_opencl) {
	cat("OpenCL support detected.\n")
	# Set the USE_OPENCL flag.
	local_cppflags <- c(local_cppflags, "-DUSE_OPENCL")
	# If OPENCL_HOME is set, use its include and lib directories.
	if (nzchar(OPENCL_HOME)) {
		local_cppflags <- c(local_cppflags, paste0("-I", file.path(OPENCL_HOME, "include")))
		local_libs     <- c(local_libs, paste0("-L", file.path(OPENCL_HOME, "lib")))
	}
	# in all our windows checking, we have to put single quotes around the paths, so that we don't get the paths broken up into different args
	if (.Platform$OS.type == "windows") {

		# If OPENCL_HOME isn't set, check for default installations.
		if (!nzchar(OPENCL_HOME)) {
			default_nvidia_opencl_path <- "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"
			default_amd_opencl_path <- "C:/Program Files (x86)/AMD APP SDK/3.0"

            # if we have openCL on default nvidia path use it.
			if (file.exists(file.path(default_nvidia_opencl_path, "include", "CL", "cl.h"))) {
				message("Using default NVIDIA OpenCL include path on Windows: ", file.path(default_nvidia_opencl_path, "include"))
				local_cppflags <- c(local_cppflags, paste0("-I'", file.path(default_nvidia_opencl_path, "include"), "'"))
				local_libs <- c(local_libs, paste0("/LIBPATH:'", file.path(default_nvidia_opencl_path, "lib", "x64"), "'"), "OpenCL")

            # try to find the include files
			} else if (file.exists(file.path(default_amd_opencl_path, "include", "CL", "cl.h"))) {
				message("Using default AMD OpenCL include path on Windows: ", file.path(default_amd_opencl_path, "include"))
				local_cppflags <- c(local_cppflags, paste0("-I'", file.path(default_amd_opencl_path, "include"), "'"))
				local_libs <- c(local_libs, paste0("/LIBPATH:'", file.path(default_amd_opencl_path, "lib"), "'"), "OpenCL")

			} else {
				warning("No default OpenCL installation found on Windows. Please set OPENCL_HOME PATH variables to use this functionality!")
			}
		} else {
			# If OPENCL_HOME is set, assume its Windows structure is valid.
			# have to make sure to wrap single quotes around stuff, else it gets broken up by the linker.
			local_cppflags <- c(local_cppflags, paste0("-I'", file.path(OPENCL_HOME, "include"), "'"))
			local_libs <- c(local_libs, paste0("/LIBPATH:'", file.path(OPENCL_HOME, "lib"), "'"), "OpenCL")
		}
	} else {
		# Non-Windows: use framework linking on macOS, standard linking on others.
		if (Sys.info()["sysname"] == "Darwin") {
			local_libs <- c(local_libs, "-framework OpenCL")
		} else {
			local_libs <- c(local_libs, "-lOpenCL")
		}
	}
}

# Write a temporary test file to try including omp.h.
test_file <- tempfile(fileext = ".c")
writeLines(c("#include <omp.h>", "int main() { return 0; }"), con = test_file)

# Try to compile the test file.
compile_result <- system(paste("R CMD SHLIB", test_file), intern = TRUE)

# Check for successful compilation by verifying the compiled shared object.
compiled_so <- sub("\\.c$", .Platform$dynlib.ext, test_file)
if (file.exists(compiled_so)) {
	cat("OpenMP header found, enabling OpenMP flags.\n")
	local_cppflags <- c(local_cppflags, "-DHAVE_OPENMP")
	if (.Platform$OS.type == "windows") {
		local_cxxflags <- c(local_cxxflags, "/openmp")
	} else if (Sys.info()["sysname"] == "Darwin") {
		# On macOS with Clang, use these flags.
		local_cxxflags <- c(local_cxxflags, "-Xpreprocessor", "-fopenmp")
		local_ldflags  <- c(local_ldflags, "-lomp")
	} else {
		local_cxxflags <- c(local_cxxflags, "-fopenmp")
		local_ldflags  <- c(local_ldflags, "-fopenmp")
	}
	file.remove(compiled_so)
} else {
	cat("omp.h not found, skipping OpenMP.\n")
}

# Clean up the temporary file.
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
