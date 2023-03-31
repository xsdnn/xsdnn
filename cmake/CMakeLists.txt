cmake_minimum_required(VERSION 3.13)
project(xsdnn CXX)


option(xsdnn_BUILD_TEST "...")
option(xsdnn_USE_DOUBLE "")
option(xsdnn_USE_DETERMENISTIC_GEN "")

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(XS_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)