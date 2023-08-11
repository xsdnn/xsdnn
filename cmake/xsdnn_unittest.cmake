add_subdirectory(${XSROOT}/cmake/external/googletest EXCLUDE_FROM_ALL)

enable_testing()

set(XSDNN_TEST_ROOT ${XSROOT}/test)

macro(AddTest TARGET SOURCES)
    add_executable(${TARGET} ${SOURCES} )
    target_link_libraries(${TARGET} GTest::gtest_main xsdnn protobuf)
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
        mmpack_v2mm_test
        ${XSDNN_TEST_ROOT}/test_sv2mm.cc
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


