//
// Created by rozhin on 13.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ONNX_LOADER_H
#define XSDNN_ONNX_LOADER_H

#include <fstream>
#include "onnx_common.h"
#include "../common/network.h"

namespace xsdnn {

class OnnxLoader {
public:
    explicit OnnxLoader(const std::string& path);
    bool BuildXsModelFromOnnx(network& net); // TODO: Impl this ...

private:
    OnnxWrapper model_;
};

} // xsdnn

#endif //XSDNN_ONNX_LOADER_H
