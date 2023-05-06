//
// Created by rozhin on 04.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "optimizer_base.h"

namespace xsdnn {

optimizer::optimizer() = default;
optimizer::optimizer(const xsdnn::optimizer &rhs) = default;
optimizer &optimizer::operator=(const xsdnn::optimizer &rhs) = default;
optimizer::~optimizer() = default;
void optimizer::reset() {}

}