add_subdirectory(${XSROOT}/cmake/external/googletest EXCLUDE_FROM_ALL)

enable_testing()

set(XSDNN_TEST_ROOT ${XSROOT}/test)

macro(AddTest TARGET SOURCES)
    add_executable(${TARGET} ${SOURCES} )
    target_link_libraries(${TARGET} GTest::gtest_main xsdnn)
    add_test(${TARGET} ${TARGET})
endmacro()

AddTest(
        mmpack_sgemm_test
        ${XSDNN_TEST_ROOT}/test_sgemm.cc
)

AddTest(
        mmpack_sv2mm_test
        ${XSDNN_TEST_ROOT}/test_sv2mm.cc
)

AddTest(
        xsdnn_fully_connected_test
        ${XSDNN_TEST_ROOT}/test_fully_connected.cc
)

