//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef MMPACK_TEST_UTILS_H
#define MMPACK_TEST_UTILS_H
#include "xsdnn.h"
#include "serializer/cerial.h"
#include <fstream>
#include <filesystem>
#include <sys/mman.h>
#include <gtest/gtest.h>
namespace fs = std::filesystem;
using namespace mmpack;
using namespace xsdnn;

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

namespace utils {

void create_directory(const std::string &directory_name) {
    fs::current_path("./");
    fs::create_directory(directory_name);
}

void init(mm_scalar* ptr, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            *ptr = i * cols + j;
            ptr += 1;
        }
    }
}

void ContainerEqual(tensor_t T1, tensor_t T2) {
    if (T1.dtype() == T2.dtype() == XsDtype::F32) {
        gsl::span<const float> T1Span = T1.GetDataAsSpan<float>();
        gsl::span<const float> T2Span = T2.GetDataAsSpan<float>();
        ASSERT_EQ(T1Span.size(), T2Span.size());
        for (size_t i = 0; i < T1Span.size(); ++i) {
            ASSERT_FLOAT_EQ(T1Span[i], T2Span[i]);
        }
    } else {
        throw xs_error("Unregonized Tensor Dtype");
    }
}

template<typename T>
void ContainerNear(tensor_t T1, tensor_t T2, T eps) {
    if (T1.dtype() == T2.dtype() == XsDtype::F32) {
        gsl::span<const float> T1Span = T1.GetDataAsSpan<float>();
        gsl::span<const float> T2Span = T2.GetDataAsSpan<float>();
        ASSERT_EQ(T1Span.size(), T2Span.size());
        for (size_t i = 0; i < T1Span.size(); ++i) {
            ASSERT_NEAR(T1Span[i], T2Span[i], eps);
        }
    } else {
        throw xs_error("Unregonized Tensor Dtype");
    }
}

template<typename T>
void vector_init(T* ptr, std::vector<T> data) {
    for (size_t i = 0; i < data.size(); ++i) {
        *ptr++ = data[i];
    }
}

void value_init(mm_scalar* ptr, mm_scalar value, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        *ptr = value;
        ptr += 1;
    }
}

void random_init(float* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        *ptr = value;
        ptr += 1;
    }
}

template<typename T>
bool cerial_testing(T& layer) {
    create_directory("layer_cerial_tmp_directory");
    std::string path = "./layer_cerial_tmp_directory/" + layer.layer_type();

    network<sequential> net_saver;
    net_saver << layer;
    net_saver.init_weight();
    net_saver.save(path);

    network<sequential> net_loader;
    net_loader.load(path);

    return net_saver == net_loader;
}

template <typename T>
class MatrixGuardBuffer {
public:
    MatrixGuardBuffer() {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void) {
        ReleaseBuffer();
    }

    T* GetBuffer(size_t Elements, bool ZeroFill = false) {
        //
        // Check if the internal buffer needs to be reallocated.
        //

        if (Elements > _ElementsAllocated) {
            ReleaseBuffer();

            //
            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.
            //

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;

            size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

            _BaseBufferSize = BytesToAllocate + GuardPadding;
            _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);


            if (_BaseBuffer == nullptr) {
                abort();
            }

            //
            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.
            //

            if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
                abort();
            }

            _ElementsAllocated = BytesToAllocate / sizeof(T);
            _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
        }

        //
        //
        //

        T* GuardAddress = _GuardAddress;
        T* buffer = GuardAddress - Elements;

        if (ZeroFill) {
            std::fill_n(buffer, Elements, T(0));

        } else {
            constexpr int MinimumFillValue = -23;
            constexpr int MaximumFillValue = 23;

            int FillValue = MinimumFillValue;
            T* FillAddress = buffer;

            while (FillAddress < GuardAddress) {
                *FillAddress++ = (T)FillValue;

                FillValue++;

                if (FillValue > MaximumFillValue) {
                    FillValue = MinimumFillValue;
                }
            }
        }

        return buffer;
    }

    void ReleaseBuffer(void) {
        if (_BaseBuffer != nullptr) {
            munmap(_BaseBuffer, _BaseBufferSize);

            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    T* _GuardAddress;
};

} // utils

#endif //MMPACK_TEST_UTILS_H
