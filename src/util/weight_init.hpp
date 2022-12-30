//
// Created by Andrei R. on 31.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_WEIGHT_INIT_HPP
#define XSDNN_WEIGHT_INIT_HPP


namespace xsdnn {

class function {
public:
    function(Scalar scale) : scale_(scale) {}
    virtual ~function() {}
    virtual void fill(Scalar* data, size_t size, size_t fan_in, size_t fan_out) = 0;

protected:
    Scalar scale_;
};

class xavier : protected function {
public:
    xavier() : function(Scalar(6.0)) {}
    explicit xavier(Scalar scale) : function(scale) {}

    virtual void fill(Scalar* data, size_t size, size_t fan_in, size_t fan_out) override {
        const Scalar bias = std::sqrt(scale_ / (fan_in + fan_out));
        uniform_rand(data, size, -bias, bias);
    }
};

} // xsdnn


#endif //XSDNN_WEIGHT_INIT_HPP
