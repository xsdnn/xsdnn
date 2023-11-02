//
// Created by rozhin on 02.11.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//
#include <utils/transpose.h>
#include <cstring>

namespace xsdnn {

size_t size_from_dimension(std::vector<size_t> ShapeVector, size_t from) {
    assert(from <= ShapeVector.size());
    size_t size = 1;
    for (size_t i = from; i < ShapeVector.size(); ++i) {
        size *= ShapeVector[i];
    }
    return size;
}

size_t size_to_dimension(std::vector<size_t> ShapeVector, size_t to) {
    assert(to <= ShapeVector.size());
    size_t size = 1;
    for (size_t i = 0; i < to; ++i) {
        size *= ShapeVector[i];
    }
    return size;
}

size_t shape_size(std::vector<size_t> XShape) {
    size_t size = 1;
    for (size_t i : XShape) {
        size *= i;
    }
    return size;
}

// TODO: make it for mmpack
template<typename T>
void ComputeInwardsSingleAxisTranspose(const T* X, T* Y, const size_t num_loops, const size_t num_readers,
                                       const size_t reads_per_loop, const size_t reads_per_reads_per_loop) {
    T* end;
    for (int64_t l = 0; l < num_loops; ++l) {
        const T* input_for_first_reader = X;

        for (auto rrpl = 0; rrpl < reads_per_reads_per_loop; ++rrpl) {
            const T* input_for_current_reader = input_for_first_reader;

            end = Y + num_readers;
            for (; Y != end;) {
                *Y++ = *input_for_current_reader;
                // skip to input position for next reader
                input_for_current_reader += reads_per_reads_per_loop;
            }

            ++input_for_first_reader;
        }

        X += reads_per_loop;
    }
}


void inwards_single_axis_transpose(mat_t* X, std::vector<size_t> XShape, mat_t* Y, xsDtype dtype, size_t from, size_t to) {
    const size_t elements_size = dtype2sizeof(dtype);
    const size_t num_loops = size_to_dimension(XShape, from);
    const size_t num_readers = XShape[from];
    const size_t block_size = size_from_dimension(XShape, to + 1);
    const size_t reads_per_loop = shape_size(XShape) / num_loops / block_size;
    const size_t reads_per_reads_per_loop = reads_per_loop / num_readers;
    const size_t bytes_per_read = block_size * elements_size;

    switch (bytes_per_read) {
        case sizeof(uint32_t):
            ComputeInwardsSingleAxisTranspose(reinterpret_cast<const uint32_t*>(GetDataRaw(X)),
                                              reinterpret_cast<uint32_t*>(GetMutableDataRaw(Y)), num_loops, num_readers,
                                              reads_per_loop, reads_per_reads_per_loop);
            break;
        default:
            xs_warning(START_MSG + "Use default implementation");
            const auto* input_data = reinterpret_cast<const uint8_t*>(GetDataRaw(X));
            auto* output_data = reinterpret_cast<uint8_t*>(GetMutableDataRaw(Y));
            // we need to use memcpy for each block
            for (int64_t l = 0; l < num_loops; ++l) {
                const uint8_t* input_for_first_reader = input_data;

                for (auto rrpl = 0; rrpl < reads_per_reads_per_loop; ++rrpl) {
                    const uint8_t* input_for_current_reader = input_for_first_reader;

                    for (int64_t r = 0; r < num_readers; ++r) {
                        std::memcpy(output_data, input_for_current_reader, bytes_per_read);
                        output_data += bytes_per_read;

                        // skip to input position for next reader
                        input_for_current_reader += (reads_per_reads_per_loop * bytes_per_read);
                    }

                    input_for_first_reader += bytes_per_read;
                }

                input_data += reads_per_loop * bytes_per_read;
            }
    }
}


void xs_single_axis_transpose(mat_t* X, std::vector<size_t> XShape, mat_t* Y, xsDtype dtype, size_t from, size_t to) {
    if (from > to) {
        throw xs_error(START_MSG + "Case from > to not implemented yet");
    } else {
        inwards_single_axis_transpose(X, XShape, Y, dtype, from, to);
    }
}

} // xsdnn