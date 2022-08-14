cmake_minimum_required(VERSION 3.22)
project(xsDNN_include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

set(CMAKE_CXX_STANDARD 17)


add_executable(test_sinus internal-test/test_sinus_fit.cpp)
add_executable(test_classification_metrics internal-test/classification-metrics-test.cpp)
add_executable(net-output-test internal-test/net-output-test.cpp)
add_executable(distribute-test internal-test/distributing-generate-test.cpp)
add_executable(save_load_net internal-test/save_load_net_test.cpp)
add_executable(new_layer_test internal-test/new_layer_test.cpp)
add_executable(bernoulli_test internal-test/bernoulli_test.cpp)
add_executable(determenistic_test internal-test/determenistic_rng_test.cpp)
add_executable(some_test internal-test/some_test.cpp)
add_executable(progress_bar internal-test/progress_bar_test.cpp)
add_executable(dropout_test internal-test/dropout_layer_test.cpp)
add_executable(bn1d_test internal-test/batchnorm1d_test.cpp)
add_executable(mnist_example example/mnist/train.cpp)
add_executable(mnist_example_test example/mnist/test.cpp)
