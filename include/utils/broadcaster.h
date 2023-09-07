//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_BROADCASTER_H
#define XSDNN_BROADCASTER_H

#include <cstdlib>
#include <vector>
#include "../gsl/span"
#include "tensor_shape.h"

namespace xsdnn {

class broadcast_iterator {
public:
    size_t current() const noexcept;
    size_t advance_by(size_t delta);
    void reserve(std::ptrdiff_t max_dims);
    void init(std::ptrdiff_t axis, std::ptrdiff_t largest);
    void allocate_counters();
    std::ptrdiff_t get_counts_front() const;
    std::ptrdiff_t get_deltas_front() const;
    void append(std::ptrdiff_t axis, std::ptrdiff_t largest);
    void stop_broadcasting();
    void start_broadcasting();

private:
    std::vector<std::ptrdiff_t> counters_;
    std::vector<std::ptrdiff_t> deltas_;
    std::vector<std::ptrdiff_t> counts_;
    std::ptrdiff_t count_{1};
    size_t index_{};
};

class broadcaster {
public:
    explicit broadcaster(shape3d s1, shape3d s2);
    size_t get_span_size() const;

private:
    std::vector<size_t> shape2vector(shape3d s);
    void vector2shape(std::vector<size_t> v);

    size_t num_dimensions(std::vector<size_t> v);

private:
    static constexpr size_t dimension_size = 3;

public:
    broadcast_iterator it1_;
    broadcast_iterator it2_;
    shape3d output_shape_;
};

class input_broadcaster {
public:
    explicit input_broadcaster(shape3d& s1, mat_t& tensor1,
                               shape3d* s2 = nullptr, mat_t* tensor2 = nullptr);

public:
    void advance_by(size_t offset);
    shape3d get_output_shape() const;
    size_t get_span_size() const;
    size_t get_input_element_size();
    bool have_two_tensors() const;

    bool input0_scalar() const;
    bool input1_scalar() const;

    void next();

    template<typename T>
    const T& GetScalar0() {
        return *(static_cast<const T*>(input0_bytes_) + broadcaster_.it1_.current());
    }

    template<typename T>
    const T& GetScalar1() {
        return *(static_cast<const T*>(input1_bytes_) + broadcaster_.it2_.current());
    }

    template<typename T>
    gsl::span<const T> GetSpan0(size_t offset, size_t num_elements) {
        return gsl::span<const T>(static_cast<const T*>(input0_bytes_) + broadcaster_.it1_.current() + offset,
                                  num_elements);
    }

    template<typename T>
    gsl::span<const T> GetSpan1(size_t offset, size_t num_elements) {
        return gsl::span<const T>(static_cast<const T*>(input1_bytes_) + broadcaster_.it2_.current() + offset,
                                  num_elements);
    }

private:
    const mat_t& input0_tensor_;
    const shape3d& input0_shape_;
    const mat_t* input1_tensor_;
    const shape3d& input1_shape_;
    const size_t input_element_size_{sizeof(mm_scalar)};
    const void* input0_bytes_{input0_tensor_.data()};
    const void* input1_bytes_{input1_tensor_ ? input1_tensor_->data(): nullptr};

    broadcaster broadcaster_{input0_shape_, input1_shape_};
    size_t span_size_{broadcaster_.get_span_size()};
};

class output_broadcaster {
public:
    explicit output_broadcaster(size_t span_size, mat_t& tensor, shape3d shape,
                                ptrdiff_t start_offset = 0, ptrdiff_t end_offset = 0);

public:
    size_t get_output_num_elements() const;
    size_t get_output_elements_size() const;
    operator bool() const;
    void next();

    template<typename T>
    gsl::span<T> GetSpanOutput(size_t offset, size_t num_elements) {
        assert(offset < span_size_ && (offset + num_elements) <= span_size_);
        return gsl::span<T>(reinterpret_cast<T*>(output_bytes_) + offset, num_elements);
    }


private:
    const size_t span_size_;
    size_t output_elements_;
    const size_t element_size_{sizeof(mm_scalar)};
    uint8_t* output_bytes_;
    const void* output_end_;
};

class broadcast {
public:
    explicit broadcast(input_broadcaster& in_broadcast,
                       output_broadcaster& out_broadcast);

public:
    bool have_two_tensors() const;

    bool input0_scalar() const;
    bool input1_scalar() const;

    void next();

    bool need_more_output();

    template <typename T>
    gsl::span<const T> GetSpanInput0() { return inputBroadcaster.GetSpan0<T>(input0_offset_, input0_num_elements_); }

    template <typename T>
    gsl::span<const T> GetSpanInput1() { return inputBroadcaster.GetSpan1<T>(input1_offset_, input1_num_elements_); }

    template <typename T>
    const T& GetScalarInput0() { return inputBroadcaster.GetScalar0<T>(); }

    template <typename T>
    const T& GetScalarInput1() { return inputBroadcaster.GetScalar1<T>(); }

    template <typename T>
    gsl::span<T> GetOutputSpan() { return outputBroadcaster.GetSpanOutput<T>(output_offset_, output_num_elements_); }

private:
    input_broadcaster inputBroadcaster;
    output_broadcaster outputBroadcaster;

    size_t input0_offset_{0};
    size_t input0_num_elements_{inputBroadcaster.get_span_size()};
    size_t input1_offset_{0};
    size_t input1_num_elements_{inputBroadcaster.get_span_size()};
    size_t output_offset_{0};
    size_t output_num_elements_{inputBroadcaster.get_span_size()};
};

using broadcast_func = void (*)(broadcast&);
struct BroadcastFuncHolder {
    broadcast_func input0scalar;
    broadcast_func input1scalar;
    broadcast_func general;
};

void BroadcastKernelLoop(broadcast& bc, BroadcastFuncHolder func_holder);

}

#endif //XSDNN_BROADCASTER_H
