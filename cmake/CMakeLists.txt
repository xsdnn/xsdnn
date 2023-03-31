cmake_minimum_required(VERSION 3.13)
project(xsdnn CXX)


option(xsdnn_BUILD_TEST OFF)
option(xsdnn_USE_DOUBLE OFF)
option(xsdnn_USE_DETERMENISTIC_GEN OFF)

if (xsdnn_USE_DOUBLE)
    add_compile_definitions(XS_USE_DOUBLE)
endif (xsdnn_USE_DOUBLE)