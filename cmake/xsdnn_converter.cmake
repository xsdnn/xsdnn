set(
        xsdnn_converter_src
        "${XSROOT_SRC}/converter/onnx_common.cc"
)

add_library(xsdnn_converter STATIC
        ${xsdnn_converter_src})