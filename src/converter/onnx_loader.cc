//
// Created by rozhin on 13.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <converter/onnx_loader.h>

namespace xsdnn {

OnnxLoader::OnnxLoader(const std::string &path) {
    std::ifstream input(path, std::ios::ate | std::ios::binary);
    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    input.read(buffer.data(), size);

    onnx::ModelProto model;
    model.ParseFromArray(buffer.data(), size);
    model_.setModelProto(model);
}

bool OnnxLoader::BuildXsModelFromOnnx(xsdnn::network &net) {

}

} // xsdnn
