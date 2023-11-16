enable_testing()

set(XSDNN_TEST_ROOT ${XSROOT}/test)

macro(AddTest TARGET SOURCES)
    add_executable(${TARGET} ${SOURCES} )
    TARGET_LINK_LIBRARIES(${TARGET} GTest::gtest_main xsdnn)

    IF(xsdnn_WITH_SERIALIZATION)
        TARGET_LINK_LIBRARIES(${TARGET} ${Protobuf_LIBRARIES})
    ENDIF()

    IF(xsdnn_BUILD_XNNPACK_ENGINE)
        TARGET_LINK_LIBRARIES(${TARGET} XNNPACK)
    ENDIF()

#    IF(xsdnn_BUILD_PROFILING_TOOLS)
#        TARGET_LINK_LIBRARIES(${TARGET} easy_profiler)
#    ENDIF()

    add_test(${TARGET} ${TARGET})
endmacro()

AddTest(
        mmpack_gemm_test
        ${XSDNN_TEST_ROOT}/test_sgemm.cc
)

#AddTest(
#        mmpack_gemm_perfomance_test
#        ${XSDNN_TEST_ROOT}/test_sgemm_perfomance.cc
#)

AddTest(
        mmpack_dot_test
        ${XSDNN_TEST_ROOT}/test_dot.cc
)

AddTest(
        mmpack_muladd_test
        ${XSDNN_TEST_ROOT}/test_muladd.cc
)

AddTest(
        mmpack_madd_test
        ${XSDNN_TEST_ROOT}/test_madd.cc
)

AddTest(
        xsdnn_fully_connected_test
        ${XSDNN_TEST_ROOT}/test_fully_connected.cc
)

AddTest(
        xsdnn_relu_test
        ${XSDNN_TEST_ROOT}/test_relu.cc
)

AddTest(
        xsdnn_network_serialization_test
        ${XSDNN_TEST_ROOT}/test_network_serialization.cc
)


AddTest(
        xsdnn_input_test
        ${XSDNN_TEST_ROOT}/test_input.cc
)

AddTest(
        xsdnn_flatten_test
        ${XSDNN_TEST_ROOT}/test_flatten.cc
)

AddTest(
        xsdnn_output_test
        ${XSDNN_TEST_ROOT}/test_output.cc
)

AddTest(
        xsdnn_max_pooling_test
        ${XSDNN_TEST_ROOT}/test_max_pooling.cc
)

AddTest(
        xsdnn_global_average_pooling_test
        ${XSDNN_TEST_ROOT}/test_global_average_pooling.cc
)

AddTest(
        xsdnn_reshape_test
        ${XSDNN_TEST_ROOT}/test_reshape.cc
)

AddTest(
        xsdnn_conv_test
        ${XSDNN_TEST_ROOT}/test_conv.cc
)

AddTest(
        xsdnn_broadcast_op_test
        ${XSDNN_TEST_ROOT}/test_broadcast.cc
)

AddTest(
        mmpack_hard_sigmoid_test
        ${XSDNN_TEST_ROOT}/test_hard_sigmoid.cc
)

AddTest(
        xsdnn_transpose_test
        ${XSDNN_TEST_ROOT}/test_transpose.cc
)

#AddTest(
#        xsdnn_profiler_test
#        ${XSDNN_TEST_ROOT}/test_profiler.cc
#)