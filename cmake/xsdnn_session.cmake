set(
        xsdnn_session_src
        "${XSROOT}/src/session/inference_session.cc"
        "${XSROOT}/src/session/inference_options.cc"
)

add_library(xsdnn_session STATIC ${xsdnn_session_src})