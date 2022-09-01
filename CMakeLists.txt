cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_NO_DTRMINIST "Set to OFF to determinist rng"           ON)
option(USE_QUIET        "Set to ON to off callback"              OFF)
option(USE_BLAS         "Set to ON to on Eigen 3 BLAS"           OFF)
option(MULTITHREADING   "Set to ON to on threading"              OFF)



option(BUILD_TESTS      "Set to ON to build tests"              ON)
option(BUILD_EXAMPLES   "Set to ON to build examples"           OFF)

set( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
if( NOT EIGEN3_INCLUDE_DIR )
    message( FATAL_ERROR
            "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()
include_directories( "${EIGEN3_INCLUDE_DIR}" )

if(USE_NO_DTRMINIST)
    add_compile_definitions(DNN_NO_DTRMINIST)
endif(USE_NO_DTRMINIST)

if(USE_QUIET)
    add_compile_definitions(DNN_BE_QUIET)
endif(USE_QUIET)

if(USE_BLAS)
    add_compile_definitions(EIGEN_USE_BLAS)
endif(USE_BLAS)


if(MULTITHREADING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
endif(MULTITHREADING)

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)

