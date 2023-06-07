set(
        xsdnn_threadpool_framework_src
        "${XSROOT_SRC}/framework/threadpool.cc"
)

add_library(xsdnn_framework STATIC
        ${xsdnn_threadpool_framework_src})