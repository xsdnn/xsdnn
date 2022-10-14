cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

option(BUILD_TESTS      "Set to ON to build tests"              OFF)


if( NOT DEFINED ENV{EIGEN3_INCLUDE_DIR} )
    message( FATAL_ERROR
            "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()
include_directories($ENV{EIGEN3_INCLUDE_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)