cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_NO_DTRMINIST "Set to OFF to determinist rng"           ON)
option(USE_QUIET        "Set to ON to off callback"              OFF)
option(MULTITHREADING   "Set to ON to on threading"              OFF)


option(BUILD_TESTS      "Set to ON to build tests"              ON)
option(BUILD_EXAMPLES   "Set to ON to build examples"           OFF)


if(USE_NO_DTRMINIST)
    add_compile_definitions(DNN_NO_DTRMINIST)
endif(USE_NO_DTRMINIST)

if(USE_QUIET)
    add_compile_definitions(DNN_BE_QUIET)
endif(USE_QUIET)


if(MULTITHREADING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
endif(MULTITHREADING)

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif(BUILD_EXAMPLES)

