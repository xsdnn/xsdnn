set(
        xsdnn_converter_src
        "${XSROOT_SRC}/converter/onnx_common.cc"
        "${XSROOT_SRC}/converter/onnx_loader.cc"
        "${XSROOT_SRC}/../include/converter/onnx.proto3.pb.cc"
)

add_library(xsdnn_converter STATIC
        ${xsdnn_converter_src})