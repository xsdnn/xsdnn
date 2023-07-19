//
// Created by rozhin on 13.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_ONNX_COMMON_H
#define XSDNN_ONNX_COMMON_H

#include "onnx.proto3.pb.h"

namespace xsdnn {

class OnnxWrapper {
public:
    OnnxWrapper();
    explicit OnnxWrapper(onnx::ModelProto& model);
    void setModelProto(onnx::ModelProto& model);

    onnx::GraphProto getModelGraph() const;
    std::string getModelGraphName() const;
    std::string getNodeOpType(int32_t i) const;
    int32_t getInitializerSize() const;
    int32_t getTensorSize(int32_t i) const;


private:
    onnx::ModelProto model_;
};

} // xsdnn

#endif //XSDNN_ONNX_COMMON_H
