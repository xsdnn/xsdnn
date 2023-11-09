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
#include <cstring>
namespace fs = std::filesystem;
using namespace mmpack;
using namespace xsdnn;

#define MAKE_MUTABLE_CONTAINER_SPAN_FP32(CONTAINER_NAME, CONTAINER_SPAN_NAME, SIZE) \
mat_t CONTAINER_NAME(SIZE * dtype2sizeof(xsdnn::kXsFloat32)); \
gsl::span<float> CONTAINER_SPAN_NAME = GetMutableDataAsSpan<float>(&CONTAINER_NAME);

#define MAKE_CONTAINER_SPAN_FP32(CONTAINER_NAME, CONTAINER_SPAN_NAME, SIZE) \
mat_t CONTAINER_NAME(SIZE * dtype2sizeof(xsdnn::kXsFloat32)); \
gsl::span<const float> CONTAINER_SPAN_NAME = GetDataAsSpan<const float>(&CONTAINER_NAME);

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

namespace utils {

void create_directory(const std::string &directory_name) {
    fs::current_path("./");
    fs::create_directory(directory_name);
}

void sequantial_init_fp32(float* ptr, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            *ptr = i * cols + j;
            ptr += 1;
        }
    }
}

void sequantial_init_fp32(mat_t& X, size_t rows, size_t cols)  {
    gsl::span<float> XSpan = GetMutableDataAsSpan<float>(&X);
    float* XSpanPtr = XSpan.data();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            *XSpanPtr = i * cols + j;
            XSpanPtr += 1;
        }
    }
}

void value_init_fp32(float* ptr, mm_scalar value, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        *ptr = value;
        ptr += 1;
    }
}

void value_init_fp32(mat_t&X, float Value) {
    gsl::span<float> XSpan = GetMutableDataAsSpan<float>(&X);
    for (size_t i = 0; i < XSpan.size() / sizeof(float); ++i) {
        XSpan[i] = Value;
    }
}

void random_init_fp32(float* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        *ptr = value;
        ptr += 1;
    }
}

void random_init_fp32(mat_t& X) {
    gsl::span<float> XSpan = GetMutableDataAsSpan<float>(&X);
    for (size_t i = 0; i < XSpan.size() / sizeof(float); ++i) {
        XSpan[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void initializer_list_init_fp32(mat_t& X, std::initializer_list<float> Value) {
    gsl::span<float> XSpan = GetMutableDataAsSpan<float>(&X);
    assert(XSpan.size() / sizeof(float) == Value.size());
    std::copy(Value.begin(), Value.end(), XSpan.begin());
}

template<typename T>
void xsAssert_eq(T lhs, T rhs, xsDtype type) {
    if (type == kXsFloat32) {
        ASSERT_FLOAT_EQ(lhs, rhs);
    } else throw xs_error("[xsAssert_eq] Unsupported dtype");
}

#ifdef XS_USE_SERIALIZATION
template<typename T>
bool cerial_testing(T& layer) {
    create_directory("layer_cerial_tmp_directory");
    std::string path = "./layer_cerial_tmp_directory/" + layer.layer_type();

    network net_saver;
    Output out;

    connect_subgraph(out, layer);
    construct_graph(net_saver, {&layer}, {&out});

    net_saver.init_weight();
    net_saver.save(path);

    network net_loader;
    net_loader.load(path);

    return net_saver == net_loader;
}
#endif

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

void
ComputeReferenceConv2D_HWC_FP32(
        size_t GroupCount,
        size_t InputChannels,
        size_t InputHeight,
        size_t InputWidth,
        size_t FilterCount,
        size_t KernelHeight,
        size_t KernelWidth,
        size_t PaddingLeftHeight,
        size_t PaddingLeftWidth,
        size_t PaddingRightHeight,
        size_t PaddingRightWidth,
        size_t DilationHeight,
        size_t DilationWidth,
        size_t StrideHeight,
        size_t StrideWidth,
        size_t OutputHeight,
        size_t OutputWidth,
        const float* X,
        const float* W,
        const float* B,
        float* Y
) {
    if (B != nullptr) {
        for (size_t oy = 0; oy < OutputHeight; oy++) {
            for (size_t ox = 0; ox < OutputWidth; ox++) {
                for (size_t g = 0; g < GroupCount; g++) {
                    for (size_t oc = 0; oc < FilterCount; oc++) {
                        Y[((oy * OutputWidth + ox) * GroupCount + g) * FilterCount + oc] =
                                B[g * FilterCount + oc];
                    }
                }
            }
        }
    } else {
        throw xs_error("Not Impl");
    }

    for (size_t oy = 0; oy < OutputHeight; oy++) {
        for (size_t ox = 0; ox < OutputWidth; ox++) {
            for (size_t ky = 0; ky < KernelHeight; ky++) {
                const size_t iy = oy * StrideHeight + ky * StrideHeight - PaddingLeftHeight;
                if (iy < InputHeight) {
                    for (size_t kx = 0; kx < KernelWidth; kx++) {
                        const size_t ix = ox * StrideWidth + kx * DilationWidth - PaddingLeftWidth;
                        if (ix < InputWidth) {
                            for (size_t g = 0; g < GroupCount; g++) {
                                for (size_t oc = 0; oc < FilterCount; oc++) {
                                    for (size_t ic = 0; ic < InputChannels; ic++) {
                                        Y[((oy * OutputWidth + ox) * GroupCount + g) * FilterCount + oc] +=
                                                X[(iy * InputWidth + ix) * GroupCount * InputChannels + g * InputChannels + ic] *
                                                W[(((g * FilterCount + oc) * KernelHeight + ky) * KernelWidth + kx) * InputChannels + ic];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // utils

#endif //MMPACK_TEST_UTILS_H
