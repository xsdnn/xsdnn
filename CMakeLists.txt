cmake_minimum_required(VERSION 3.22)
project(xsDNN_include)

set(CMAKE_CXX_STANDARD 17)

add_executable(test_sinus internal-test/test_sinus_fit.cpp)
add_executable(test_classification_metrics internal-test/classification-metrics-test.cpp)
add_executable(net-output-test internal-test/net-output-test.cpp)
add_executable(distribute-test internal-test/distributing-generate-test.cpp)
add_executable(save_load_net internal-test/save_load_net_test.cpp)



