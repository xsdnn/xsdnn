cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN-testing VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        edge_node_test
        edge_node_test.cpp
)
target_link_libraries(
        edge_node_test
        GTest::gtest_main
)

##########################################################

add_executable(
        colored_print_visual_test
        colored_print_visual_test.cpp
)
target_link_libraries(
        colored_print_visual_test
        GTest::gtest_main
)

##########################################################

add_executable(
        serialization_helper_test
        serialization_helper_test.cpp
)
target_link_libraries(
        serialization_helper_test
        GTest::gtest_main
)

##########################################################

add_executable(
        matmul_op_test
        matmul_op_test.cpp
)
target_link_libraries(
        matmul_op_test
        GTest::gtest_main
)

##########################################################


include(GoogleTest)
