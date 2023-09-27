//
// Created by rozhin on 19.07.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/add.h>
#include <algorithm>
#include <utils/macro.h>

namespace xsdnn {

std::vector<shape3d> add::in_shape() const {
    return std::vector<shape3d>(n_input_, shape_);
}

std::vector<shape3d> add::out_shape() const {
    return {shape_};
}

std::string add::layer_type() const {
    return "add";
}

void add::forward_propagation(const std::vector<BTensor *> &in_data,
                              std::vector<BTensor *> &out_data) {
    const BTensor& In = *in_data[0];
    BTensor& Out = *out_data[0];

    for (size_t sample = 0; sample < In.size(); ++sample) {
        gsl::span<const float> InSpan = In[sample].GetDataAsSpan<float>();
        gsl::span<float> OutSpan = Out[sample].GetMutableDataAsSpan<float>();
        std::copy(InSpan.begin(), InSpan.end(), OutSpan.begin());
    }

    for (size_t sample = 0; sample < In.size(); ++sample) {
        for (size_t i = 1; i < n_input_; i++) {
            gsl::span<const float> in_sample = (*in_data[i])[sample].GetDataAsSpan<float>();
            gsl::span<float> out_sample = Out[sample].GetMutableDataAsSpan<float>();
            std::transform(in_sample.begin(),
                           in_sample.end(),
                           out_sample.begin(),
                           out_sample.begin(),
                           [](float x, float y){ return x + y; });
        }
    }
}


} // xsdnn