set(
        xsdnn_serializer_src
        "${XSROOT}/include/serializer/xs.proto3.pb.cc"
        "${XSROOT_SRC}/layers/layer_register.cc"
)

add_library(xsdnn_serializer STATIC ${xsdnn_serializer_src})