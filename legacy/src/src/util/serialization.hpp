//
// Created by Andrei R. on 30.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_SERIALIZATION_HPP
#define XSDNN_SERIALIZATION_HPP

#include "../layers/layer.hpp"
#include "serialization_helper.hpp"

namespace xsdnn {
struct cerial {
    static inline void serialize(json& /*meta*/, layer& /*layer*/) {
        /*
         * meta["in_size"] = layer.in_size;
         * meta["out_size"] = layer.out_size;
         * meta["has_bias"] = layer.has_bias;
         */

        /*
         * io::save_weight(layer.w, layer.b); // Это должно быть реализовано через variadic templates
         */
    }

    static inline void deserialize(json& /*meta*/, layer& /*layer*/) {
        /*
         * layer.in_size = meta["in_size"];
         * layer.out_size = meta["out_size"];
         * layer.has_bias = meta["has_bias"];
         */

        /*
         * io::load_weight(layer.w, layer.b) // Также реализовано через variadic templates
         */
    }
};
} // xsdnn


#endif //XSDNN_SERIALIZATION_HPP
