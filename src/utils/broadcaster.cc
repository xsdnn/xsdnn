//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <utils/broadcaster.h>
#include <utils/xs_error.h>
#include <algorithm>

namespace xsdnn {

size_t broadcast_iterator::current() const noexcept {
    return index_;
}

size_t broadcast_iterator::advance_by(size_t delta) {
    size_t index = index_;

    index_ += deltas_[0] * delta;
    counters_[0] += delta;
    if (counters_[0] == counts_[0]) {
        counters_[0] = 0;
        for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
            index_ += deltas_[counterIndex];
            if (++counters_[counterIndex] != counts_[counterIndex])
                break;
            counters_[counterIndex] = 0;
        }
    } else if (counters_[0] > counts_[0]) {  // Keep original logic above so that in most case it is faster
        delta = counters_[0] / counts_[0];
        counters_[0] = counters_[0] % counts_[0];
        for (size_t counterIndex = 1; counterIndex < counters_.size(); counterIndex++) {
            index_ += delta * deltas_[counterIndex];
            counters_[counterIndex] += delta;
            if (counters_[counterIndex] < counts_[counterIndex]) break;
            delta = counters_[counterIndex] / counts_[counterIndex];
            counters_[counterIndex] = counters_[counterIndex] % counts_[counterIndex];
        }
    }
    return index;
}

void broadcast_iterator::reserve(std::ptrdiff_t max_dims) {
    deltas_.reserve(static_cast<size_t>(max_dims));
    counts_.reserve(static_cast<size_t>(max_dims));
}

void broadcast_iterator::init(std::ptrdiff_t axis, std::ptrdiff_t largest) {
    if (!(axis == 1 || axis == largest)) {
        throw xsdnn::xs_error("Attempting to broadcast an axis by a dimension other than 1. "
                        + std::to_string(axis) + " by " + std::to_string(largest));
    }

    deltas_.push_back(axis > 1);
    counts_.push_back(largest);
    count_ *= axis;
}

void broadcast_iterator::allocate_counters() {
    counters_.resize(counts_.size(), 0);
}

std::ptrdiff_t broadcast_iterator::get_counts_front() const {
    return counts_.front();
}

std::ptrdiff_t broadcast_iterator::get_deltas_front() const {
    return deltas_.front();
}

void broadcast_iterator::append(std::ptrdiff_t axis, std::ptrdiff_t largest) {
    if (!(axis == 1 || axis == largest)) {
        throw xsdnn::xs_error("Attempting to broadcast an axis by a dimension other than 1. "
        + std::to_string(axis) + " by " + std::to_string(largest));
    }

    // If we're greater than 1, it doesn't matter what the other tensor does
    if (axis > 1) {
        if (deltas_.back() <= 0)  // Were we broadcasting
            stop_broadcasting();
    } else {  // We must be 1, at this point
        if (deltas_.back() > 0)
            start_broadcasting();
    }

    counts_.back() *= largest;  // Just increase the last count
    count_ *= axis;
}

void broadcast_iterator::stop_broadcasting() {
    deltas_.push_back(count_);
    counts_.push_back(1);
}

void broadcast_iterator::start_broadcasting() {
    deltas_.push_back(-count_);
    counts_.push_back(1);
}

std::vector<size_t> broadcaster::shape2vector(xsdnn::shape3d s) {
    assert(s.C >= 1 && s.H >= 1 && s.W >= 1);
    return {s.C, s.H, s.W};
}

size_t broadcaster::num_dimensions(std::vector<size_t> v) {
    return 3 - std::count(v.begin(), v.end(), 1);
}

broadcaster::broadcaster(xsdnn::shape3d s1, xsdnn::shape3d s2) {
    std::vector<size_t> shape1 = shape2vector(s1);
    std::vector<size_t> shape2 = shape2vector(s2);
    std::vector<size_t> out_shape(dimension_size);

    auto iter1 = shape1.end();
    auto iter2 = shape2.end();
    auto output_shape = out_shape.end();

    size_t index = 0;
    for (; index < dimension_size; index++) {
        ptrdiff_t axis1 = static_cast<ptrdiff_t>(*--iter1);
        ptrdiff_t axis2 = static_cast<ptrdiff_t>(*--iter2);

        ptrdiff_t largest = std::max<ptrdiff_t>(axis1, axis2);
        ptrdiff_t smallest = std::min<ptrdiff_t>(axis1, axis2);
        ptrdiff_t dim_to_use = largest;

        if (smallest == 0) {
            if (largest > 1) {
                throw xsdnn::xs_error("Can broadcast 0 by 0 or 1. " + std::to_string(largest) + " is invalid.");
            }
            dim_to_use = smallest;
        }

        *--output_shape = dim_to_use;

        // if both 1, or a 1 and 0, and there are more dims, we can let the next iteration do the Init
        if (dim_to_use <= 1 && index + 1 < dimension_size)
            continue;

        it1_.init(axis1, dim_to_use);
        it2_.init(axis2, dim_to_use);
        index++;
        break;
    }

    for (; index < dimension_size; index++) {
        ptrdiff_t axis1 = static_cast<ptrdiff_t>(*--iter1);
        ptrdiff_t axis2 = static_cast<ptrdiff_t>(*--iter2);

        ptrdiff_t largest = std::max(axis1, axis2);
        ptrdiff_t smallest = std::min(axis1, axis2);
        ptrdiff_t dim_to_use = largest;

        if (smallest == 0) {
            if (largest > 1) {
                throw xsdnn::xs_error("Can broadcast 0 by 0 or 1. " + std::to_string(largest) + " is invalid.");
            }
            dim_to_use = smallest;
        }

        *--output_shape = dim_to_use;

        if (largest == 1)  // Nothing to do in this case
            continue;

        it1_.append(axis1, dim_to_use);
        it2_.append(axis2, dim_to_use);
    }

    // If one shape is bigger than another we need to broadcast the smaller onto the bigger from this point on
    for (; index < dimension_size; index++) {
        if (dimension_size == shape2.size()) {
            ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter2);
            it1_.append(1, axis);
            it2_.append(axis, axis);
            *--output_shape = axis;
        } else {
            ptrdiff_t axis = static_cast<ptrdiff_t>(*--iter1);
            it1_.append(axis, axis);
            it2_.append(1, axis);
            *--output_shape = axis;
        }
    }

    output_shape_.reshape(output_shape[0], output_shape[1], output_shape[2]);

    it1_.allocate_counters();
    it2_.allocate_counters();
}

size_t broadcaster::get_span_size() const {
    return std::min(it1_.get_counts_front(), it2_.get_counts_front()); // FIXME: прокинуть dtype
}

input_broadcaster::input_broadcaster(xsDtype dtype,
                                     xsdnn::shape3d &s1,
                                     xsdnn::mat_t &tensor1,
                                     xsdnn::shape3d *s2,
                                     xsdnn::mat_t *tensor2)
    : dtype_(dtype), input0_tensor_(tensor1), input0_shape_(s1), input1_tensor_(tensor2), input1_shape_(*s2) {}

void input_broadcaster::advance_by(size_t offset) {
    if (offset % span_size_) {
        throw xs_error("InputBroadcaster can only start at span boundary!");
    }
    broadcaster_.it1_.advance_by(offset);
    broadcaster_.it2_.advance_by(offset);
}

shape3d input_broadcaster::get_output_shape() const {
    return broadcaster_.output_shape_;
}

size_t input_broadcaster::get_span_size() const {
    return span_size_;
}

size_t input_broadcaster::get_input_element_size() {
    return input_element_size_;
}

bool input_broadcaster::have_two_tensors() const {
    return input1_tensor_ != nullptr;
}

bool input_broadcaster::input0_scalar() const {
    return broadcaster_.it1_.get_deltas_front() == 0;
}

bool input_broadcaster::input1_scalar() const {
    return broadcaster_.it2_.get_deltas_front() == 0;
}

void input_broadcaster::next() {
    advance_by(span_size_);
}

output_broadcaster::output_broadcaster(xsDtype dtype, size_t span_size, xsdnn::mat_t &tensor, xsdnn::shape3d shape,
                                       ptrdiff_t start_offset, ptrdiff_t end_offset)
    : dtype_(dtype), span_size_(span_size) {
    ptrdiff_t len = shape.size();
    ptrdiff_t real_end = (end_offset <= 0) ? len : end_offset;
    if (start_offset != 0 || end_offset != 0) {  // Keep original semantic
        throw xs_error("[output_broadcasted] NotImplementedYet");
    }
    output_elements_ = real_end - start_offset;
    output_bytes_ = reinterpret_cast<uint8_t*>(tensor.data()) + (start_offset * element_size_);
    output_end_ = output_bytes_ + ((real_end - start_offset) * element_size_);
}

size_t output_broadcaster::get_output_element_size() const {
    return element_size_;
}

size_t output_broadcaster::get_output_num_elements() const {
    return output_elements_;
}

output_broadcaster::operator bool() const {
    return output_bytes_ != output_end_;
}

void output_broadcaster::next() {
    output_bytes_ += (span_size_ * element_size_);
}

broadcast::broadcast(xsdnn::input_broadcaster &in_broadcast, xsdnn::output_broadcaster &out_broadcast)
    : inputBroadcaster(in_broadcast), outputBroadcaster(out_broadcast) {}

bool broadcast::have_two_tensors() const {
    return inputBroadcaster.have_two_tensors();
}

bool broadcast::input0_scalar() const {
    return inputBroadcaster.input0_scalar();
}

bool broadcast::input1_scalar() const {
    return inputBroadcaster.input1_scalar();
}

void broadcast::next() {
    inputBroadcaster.next();
    outputBroadcaster.next();
}

bool broadcast::need_more_output() {
    return outputBroadcaster;
}

void BroadcastKernelLoop(broadcast& bc, BroadcastFuncHolder func_holder) {
    if (!bc.have_two_tensors()) {
        throw xs_error("[BroadcastKernelLoop] 2 tensors are not available");
    }
    if (bc.input0_scalar()) {
        while(bc.need_more_output()) {
            func_holder.input0scalar(bc);
            bc.next();
        }
    } else if (bc.input1_scalar()) {
        while(bc.need_more_output()) {
            func_holder.input1scalar(bc);
            bc.next();
        }
    } else {
        while(bc.need_more_output()) {
            func_holder.general(bc);
            bc.next();
        }
    }
}

} // xsdnn
