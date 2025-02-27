cmake_minimum_required(VERSION 3.30)
project(KNN_PACKAGE LANGUAGES CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Collect base source files.
set(SRC_FILES
        library.cpp
        CPP_KNN.cpp
)

# Conditionally add CUDA sources.
find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "CUDA found: including CUDA implementation.")
    list(APPEND SRC_FILES run.cu)
    add_definitions(-DHAVE_CUDA)
else()
    message(STATUS "CUDA not found: skipping CUDA implementation.")
endif()

# Conditionally add OpenCL sources.
find_package(OpenCL)
if (OpenCL_FOUND)
    message(STATUS "OpenCL found: including OpenCL implementation.")
    list(APPEND SRC_FILES openCL_KNN.cpp)
    add_definitions(-DHAVE_OPENCL)
else()
    message(STATUS "OpenCL not found: skipping OpenCL implementation.")
endif()

# Create the shared library target.
add_library(KNN_PACKAGE SHARED ${SRC_FILES})
set_target_properties(KNN_PACKAGE PROPERTIES OUTPUT_NAME "CWUKNN")
set_target_properties(KNN_PACKAGE PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../libs"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../libs"
)
if(WIN32)
    set_target_properties(KNN_PACKAGE PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(KNN_PACKAGE PROPERTIES SUFFIX ".so")
endif()

# Conditionally enable OpenMP.
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling parallelization.")
    add_definitions(-DHAVE_OPENMP)
    target_link_libraries(KNN_PACKAGE PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found: proceeding without OpenMP.")
endif()

# R Integration: Retrieve and set R's include directory.
execute_process(
        COMMAND R CMD config --cppflags
        OUTPUT_VARIABLE R_CPPFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "-I" "" R_INCLUDE_DIR "${R_CPPFLAGS}")
string(REGEX REPLACE " .*$" "" R_INCLUDE_DIR "${R_INCLUDE_DIR}")
message(STATUS "Using R include directory: ${R_INCLUDE_DIR}")
target_include_directories(KNN_PACKAGE PRIVATE ${R_INCLUDE_DIR})

# R Integration: Retrieve and apply R's linker flags.
execute_process(
        COMMAND R CMD config --ldflags
        OUTPUT_VARIABLE R_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Original R ldflags: ${R_LDFLAGS}")
# Remove problematic flags.
string(REPLACE "-lemutls_w" "" R_LDFLAGS "${R_LDFLAGS}")
string(REPLACE "-lheapt_w" "" R_LDFLAGS "${R_LDFLAGS}")
message(STATUS "Modified R ldflags: ${R_LDFLAGS}")
separate_arguments(R_LDFLAGS)
target_link_libraries(KNN_PACKAGE PRIVATE ${R_LDFLAGS})

# If OpenCL was found, add its include directories and link its framework.
if (OpenCL_FOUND)
    target_include_directories(KNN_PACKAGE PRIVATE ${OpenCL_INCLUDE_DIRS}
            /System/Library/Frameworks/OpenCL.framework/Headers)
    target_link_libraries(KNN_PACKAGE PRIVATE "-framework OpenCL")
endif()

# Installation rule.
install(TARGETS KNN_PACKAGE LIBRARY DESTINATION "libs")