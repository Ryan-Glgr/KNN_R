/**
 * @file: library.h
 * @brief: Defines the functions created for internal use provided specific compile conditions
 *
 * @author: Ryan Gallagher
 * @date: 27.2.2025
 *
 */
#ifndef LIBRARY_H
#define LIBRARY_H
#include <Rcpp.h>
#pragma once


#ifdef USE_CUDA
#pragma message("library.h: Compiling WITH CUDA FUNCTIONALITY!")

// if we are on windows its windows.h to load dlls. itt's dlfcn.h on unix based for .so files.
#ifdef _WIN32
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif
typedef double (*cudaKNN_fptr)(double*, int, double*, int, int);

inline double cudaKNN(double* vec1, int size1, double* vec2, int size2, int k) {
    // We use static variables so we only resolve symbols once
    // and not on every call to cudaKNN().
    static bool initialized = false;
    static cudaKNN_fptr func = nullptr;

    if (!initialized) {
      Rcpp::Function systemFile("system.file");

#ifdef _WIN32
      // Windows .dll
      Rcpp::String libPath = systemFile("kernels", "cudaKNN.dll",
                                        Rcpp::Named("package", "CWUKNN"));
      if (libPath.get_cstring() == nullptr || std::string(libPath.get_cstring()).empty()) {
        Rcpp::stop("Could not locate cudaKNN.dll via system.file in inst/kernels.");
      }

      HMODULE hModule = LoadLibraryA(libPath.get_cstring());
      if (!hModule) {
        Rcpp::stop("Failed to load CUDA DLL at: %s", libPath.get_cstring());
      }

      func = (cudaKNN_fptr)GetProcAddress(hModule, "cudaKNN");
      if (!func) {
        FreeLibrary(hModule);
        Rcpp::stop("Failed to get 'cudaKNN' symbol from cudaKNN.dll");
      }

#else
      // Unix-like .so
      Rcpp::String libPath = systemFile("kernels", "cudaKNN.so",
                                        Rcpp::Named("package", "CWUKNN"));
      if (libPath.get_cstring() == nullptr || std::string(libPath.get_cstring()).empty()) {
        Rcpp::stop("Could not locate cudaKNN.so via system.file in inst/kernels.");
      }

      void* handle = dlopen(libPath.get_cstring(), RTLD_LAZY);
      if (!handle) {
        Rcpp::stop("Failed to load CUDA .so at: %s\nError: %s",
                   libPath.get_cstring(), dlerror());
      }

      func = (cudaKNN_fptr)dlsym(handle, "cudaKNN");
      if (!func) {
        dlclose(handle);
        Rcpp::stop("Failed to get 'cudaKNN' symbol from cudaKNN.so");
      }
#endif

      // Mark that we've successfully loaded the function pointer.
      initialized = true;
    }
    // The interface is the same as our stub below.
    return func(vec1, size1, vec2, size2, k);
}

#else
#pragma message("library.h: Compiling WITHOUT CUDA FUNCTIONALITY!")

// this is the implementation we use if USE_CUDA Isn't set. if we don't have cuda we use the NO SUPPORT one. if we did, we use above.
inline double cudaKNN(double* vec1, int size1, double* vec2, int size2, int k) {
    // Stub implementation
    Rcpp::stop("CUDA support is not available on this system.");
}

#endif // end of use cuda flag section.

// OpenCL version
#ifdef USE_OPENCL
  #pragma message("library.h: Compiling WITH openCL FUNCTIONALITY!")
  extern double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type);
#else
  #pragma message("library.h: Compiling WITHOUT openCL FUNCTIONALITY!")
  inline double openCL(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int K, const std::string &type) {
      Rcpp::stop("OpenCL support is not available on this machine.");
}
#endif

// Pure C++ implementation declaration. not conditional since it's the default.
extern float IE_xy(Rcpp::NumericVector data_x, Rcpp::NumericVector data_y, int k);

#endif

