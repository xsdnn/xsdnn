cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(xsDNN-testing)

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
        all_test
        test.cpp
)
target_link_libraries(
        all_test
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(all_test)
