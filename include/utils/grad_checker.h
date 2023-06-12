//
// Created by rozhin on 08.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GRAD_CHECKER_H
#define XSDNN_GRAD_CHECKER_H

#include <layers/layer.h>

namespace xsdnn {

class GradChecker {
public:
    enum class mode { full, random };
    enum class status { ok, bad };

public:
    GradChecker(layer* l_ptr, mode m);
    GradChecker(const GradChecker&);
    GradChecker& operator=(const GradChecker&);

public:
    status run();

private:
    mm_scalar numeric_gradient(layer* l_ptr,
                          std::vector<tensor_t> in_data,
                          std::vector<tensor_t> out_data,
                          const size_t in_concept_idx,
                          const size_t out_concept_idx,
                          const size_t in_position_idx,
                          const size_t out_position_idx);

    mm_scalar analytical_gradient(layer* l_ptr,
                                  std::vector<tensor_t> in_data,
                                  std::vector<tensor_t> out_data,
                                  std::vector<tensor_t> in_grad,
                                  std::vector<tensor_t> out_grad,
                                  const size_t in_concept_idx,
                                  const size_t out_concept_idx,
                                  const size_t in_position_idx,
                                  const size_t out_position_idx);

public:
    layer* l_ptr_;
    mode mode_;
};

} // xsdnn

#endif //XSDNN_GRAD_CHECKER_H
