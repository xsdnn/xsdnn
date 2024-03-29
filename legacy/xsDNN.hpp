//
// Created by Andrei R. on 13.10.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_XSDNN_HPP
#define XSDNN_XSDNN_HPP
#define EIGEN_USE_THREADS

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "src/config.hpp"

#include "src/util/math.hpp"
#include "src/util/color_print.hpp"
#include "src/util/xs_error.hpp"
#include "src/util/utils.hpp"
#include "src/util/serialization_helper.hpp"
#include "src/util/serialization.hpp"
#include "src/util/rng.hpp"
#include "src/util/random.hpp"
#include "src/util/weight_init.hpp"

#include "src/node.hpp"
#include "src/core/backend.hpp"
#include "src/layers/layer.hpp"
#endif //XSDNN_XSDNN_HPP
