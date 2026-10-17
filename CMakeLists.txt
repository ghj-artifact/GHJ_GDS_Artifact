cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures to build, for example 90 or 90;120")
endif()

project(GHJ_GDS_Artifact LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

find_path(GHJ_CUFILE_INCLUDE_DIR
    NAMES cufile.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    REQUIRED)

find_library(GHJ_CUFILE_LIBRARY
    NAMES cufile
    HINTS
        "${CUDAToolkit_LIBRARY_DIR}"
        "${CUDAToolkit_LIBRARY_ROOT}/lib64"
    REQUIRED)

add_library(ghj_cuda_dependencies INTERFACE)
target_include_directories(ghj_cuda_dependencies
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/Data_Generate"
        "${CMAKE_CURRENT_SOURCE_DIR}/GHJ_GDS"
        "${CMAKE_CURRENT_SOURCE_DIR}/GHJ_GDS/Partitioner"
        "${CMAKE_CURRENT_SOURCE_DIR}/GHJ_GDS/Joiner"
        "${GHJ_CUFILE_INCLUDE_DIR}")
target_link_libraries(ghj_cuda_dependencies
    INTERFACE
        CUDA::cudart
        Threads::Threads
        "${GHJ_CUFILE_LIBRARY}")

function(ghj_add_cuda_library target source_file)
    add_library(${target} STATIC "${source_file}")
    target_link_libraries(${target} PUBLIC ghj_cuda_dependencies)
endfunction()

ghj_add_cuda_library(ghj_data_generator
    Data_Generate/gpu_data_generator.cu)
ghj_add_cuda_library(ghj_partitioner_wps
    GHJ_GDS/Partitioner/gpu_partitioner_wps.cu)
ghj_add_cuda_library(ghj_partitioner_smh
    GHJ_GDS/Partitioner/gpu_partitioner_smh.cu)
ghj_add_cuda_library(ghj_joiner
    GHJ_GDS/Joiner/gpu_hash_join.cu)

add_executable(GHJ_GDS_WPS GHJ_GDS/GHJ_GDS_WPS.cu)
target_link_libraries(GHJ_GDS_WPS
    PRIVATE
        ghj_data_generator
        ghj_partitioner_wps
        ghj_joiner)

add_executable(GHJ_GDS_SMH GHJ_GDS/GHJ_GDS_SMH.cu)
target_link_libraries(GHJ_GDS_SMH
    PRIVATE
        ghj_data_generator
        ghj_partitioner_smh
        ghj_joiner)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/GHJ_GDS/cufile_gds4m.json"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cufile_gds4m.json"
    COPYONLY)

message(STATUS "cuFile header: ${GHJ_CUFILE_INCLUDE_DIR}")
message(STATUS "cuFile library: ${GHJ_CUFILE_LIBRARY}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
