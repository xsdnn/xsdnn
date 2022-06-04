cmake_minimum_required(VERSION 3.22)
project(xsDNN_include)

set(CMAKE_CXX_STANDARD 14)

# start mnist parse
find_package(MNIST PATHS .)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})
add_executable(mnist_example neuralnetwork/internal-test/mnist_dataset_test.cpp)
target_compile_features(mnist_example PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
# end mnist parse

add_executable(test_sinus neuralnetwork/internal-test/test_sinus_fit.cpp)
add_executable(test_classification_metrics neuralnetwork/internal-test/classification-metrics-test.cpp)

