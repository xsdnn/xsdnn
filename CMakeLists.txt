cmake_minimum_required(VERSION 3.22)
project(xsDNN_include)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_sinus neuralnetwork/internal-test/test_sinus_fit.cpp)
add_executable(test_classification_metrics neuralnetwork/internal-test/classification-metrics-test.cpp)
