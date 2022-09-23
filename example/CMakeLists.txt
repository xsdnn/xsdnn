cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN-example VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(example_mnist_train      mnist/train.cpp)
add_executable(example_mnist_test       mnist/test.cpp)
add_executable(example_sinus            approximate_sin/main.cpp)
