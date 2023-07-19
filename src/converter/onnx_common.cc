//
// Created by rozhin on 13.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <converter/onnx_common.h>

namespace xsdnn {

OnnxWrapper::OnnxWrapper(onnx::ModelProto &model) : model_(model) {}
OnnxWrapper::OnnxWrapper() {}

void OnnxWrapper::setModelProto(onnx::ModelProto &model) {
    model_ = model;
}

onnx::GraphProto OnnxWrapper::getModelGraph() const {
    return model_.graph();
}

std::string OnnxWrapper::getModelGraphName() const {
    return model_.graph().name();
}

int32_t OnnxWrapper::getInitializerSize() const {
    return model_.graph().initializer_size();
}

std::string OnnxWrapper::getNodeOpType(int32_t i) const {
    return model_.graph().node(i).op_type();
}

int32_t OnnxWrapper::getTensorSize(int32_t i) const {
    int32_t size = 1;
    for (size_t j = 0; j < model_.graph().initializer(i).dims_size(); ++j) {
        size *= model_.graph().initializer(i).dims(j);
    }
    return size;
}

} // xsdnn