cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_DOUBLE      "Set to ON to use double. Default float" OFF)
option(NO_DTRMNST      "Set to ON to use not fixed seed in rng" OFF)
option(BUILD_TEST      "Set to ON to build tests"               ON)
option(BUILD_EXAMPLE   "Set to ON to build examples"	        OFF)


if( NOT DEFINED ENV{EIGEN3_INCLUDE_DIR} )
    message( FATAL_ERROR
            "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

include_directories($ENV{EIGEN3_INCLUDE_DIR})
include_directories(json/single_include)
include_directories(spdlog/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -march=native")
set(CMAKE_CXX_FLAGS "-Wall -pedantic -Werror -Wextra")


if(BUILD_TEST)
    add_subdirectory(test)
endif(BUILD_TEST)

if(USE_DOUBLE)
    add_compile_definitions(DNN_USE_DOUBLE)
endif(USE_DOUBLE)

if(NO_DTRMNST)
    add_compile_definitions(DNN_NO_DTRMNST)
endif(NO_DTRMNST)

add_executable(main main.cpp)