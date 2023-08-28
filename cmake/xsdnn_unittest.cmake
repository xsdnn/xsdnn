add_subdirectory(${XSROOT}/cmake/external/googletest EXCLUDE_FROM_ALL)

enable_testing()

set(XSDNN_TEST_ROOT ${XSROOT}/test)

macro(AddTest TARGET SOURCES)
    add_executable(${TARGET} ${SOURCES} )
    target_link_libraries(${TARGET} GTest::gtest_main xsdnn protobuf absl_log_internal_message absl_log_internal_check_op)
    add_test(${TARGET} ${TARGET})
endmacro()

AddTest(
        mmpack_gemm_test
        ${XSDNN_TEST_ROOT}/test_sgemm.cc
)

AddTest(
        mmpack_gemm_perfomance_test
        ${XSDNN_TEST_ROOT}/test_sgemm_perfomance.cc
)

AddTest(
        mmpack_dot_test
        ${XSDNN_TEST_ROOT}/test_dot.cc
)

AddTest(
        mmpack_muladd_test
        ${XSDNN_TEST_ROOT}/test_muladd.cc
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
        xsdnn_batch_norm_test
        ${XSDNN_TEST_ROOT}/test_batch_norm.cc
)

AddTest(
        xsdnn_input_test
        ${XSDNN_TEST_ROOT}/test_input.cc
)

AddTest(
        xsdnn_abs_test
        ${XSDNN_TEST_ROOT}/test_abs.cc
)

AddTest(
        xsdnn_acos_test
        ${XSDNN_TEST_ROOT}/test_acos.cc
)

AddTest(
        xsdnn_add_test
        ${XSDNN_TEST_ROOT}/test_add.cc
)

AddTest(
        xsdnn_and_test
        ${XSDNN_TEST_ROOT}/test_and.cc
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

