cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -pedantic -Werror -Wextra")

option(BUILD_TEST      "Set to ON to build tests"               ON)
option(BUILD_EXAMPLE   "Set to ON to build examples"	        OFF)
option(USE_DOUBLE      "Set to ON to use double. Default float" OFF)

if( NOT DEFINED ENV{EIGEN3_INCLUDE_DIR} )
    message( FATAL_ERROR
            "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

include_directories($ENV{EIGEN3_INCLUDE_DIR})
include_directories(third_party/json/single_include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

if(BUILD_TEST)
    add_subdirectory(test)
endif(BUILD_TEST)

if(USE_DOUBLE)
    add_compile_definitions(DNN_USE_DOUBLE)
endif(USE_DOUBLE)
