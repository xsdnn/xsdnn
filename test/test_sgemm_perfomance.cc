//
// Created by rozhin on 26.06.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include "test_utils.h"
#include <iostream>
using namespace mmpack;

bool check_eq(float x1, float x2, float eps) {
    return std::abs(x1 - x2) < eps;
}

struct mnk_holder {
    mnk_holder(size_t m, size_t n, size_t k, float gemm, float ref, float diff)
        : M(m), N(n), K(k), GEMM(gemm), Ref(ref), Diff(diff) {}
    size_t M;
    size_t N;
    size_t K;
    float GEMM;
    float Ref;
    float Diff;
};

class SGemmTester {
public:
    SGemmTester() = default;

    void ExecuteShort() {
        constexpr size_t M = 32;
        constexpr size_t N = 32;
        constexpr size_t K = 32;
        std::vector<mnk_holder> bad_result;

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    init(m, n, k);
                    MmGemm(
                            mmpack::CblasNoTrans,
                            mmpack::CblasNoTrans,
                            m, n, k,
                            1.0f,
                            A_.data(), k,
                            B_.data(), n,
                            0.0f,
                            C_.data(), n);

                    ReferenceGemm(
                            mmpack::CblasNoTrans,
                            mmpack::CblasNoTrans,
                            m, n, k,
                            1.0f,
                            A_.data(), k,
                            B_.data(), n,
                            0.0f,
                            CReference.data(), n
                            );


                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            if (!check_eq(CReference[i * n + j], C_[i * n + j], std::numeric_limits<mm_scalar>::epsilon())) {
                                bad_result.push_back(mnk_holder(m, n, k, C_[i * n + j], CReference[i * n + j],
                                                                std::abs(C_[i * n + j] - CReference[i * n + j])));
                            }
                        }
                    }

                    clear();
                }
            }
        }

        if (bad_result.size() != 0) {
            for (size_t i = 0; i < bad_result.size(); ++i) {
                std::cout << "\x1B[31m" << "Critical Error Mismatch!" << "\x1B[31m" << std::endl;
                std::cout << "\x1B[33m" << "M: " << bad_result[i].M << " N: " << bad_result[i].N << " K: " << bad_result[i].K <<
                          "\tGEMM: " << bad_result[i].GEMM << "\nReference: " << bad_result[i].Ref <<
                          "\nDiff: " << bad_result[i].Diff  << "\x1B[33m" << "\n\n\n";
            }
        } else {
            std::cout << "\x1B[32m" << "All test passed!" << "\x1B[32m" << std::endl;
        }
    }

    void ExecuteLong() {

    }

    void ReferenceGemm(
            CBLAS_TRANSPOSE TransA,
            CBLAS_TRANSPOSE TransB,
            size_t M,
            size_t N,
            size_t K,
            float alpha,
            const float* A,
            size_t lda,
            const float* B,
            size_t ldb,
            float beta,
            float* C,
            size_t ldc
    ) {
        if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
            if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        const float* a = A + (m * lda);
                        const float* b = B + n;
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; ++k) {
                            sum += (*a * *b);
                            b += ldb;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            } else {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        const float* a = A + (m * lda);
                        const float* b = B + (n * ldb);
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; ++k) {
                            sum += (*a * *b);
                            a += 1;
                            b += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }
        } else {
            if (TransB == CBLAS_TRANSPOSE::CblasNoTrans) {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        const float* a = A + m;
                        const float* b = B + n;
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; ++k) {
                            sum += (*a * *b);
                            a += lda;
                            b += ldb;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            } else {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        const float* a = A + m;
                        const float* b = B + n * ldb;
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; ++k) {
                            sum += (*a * *b);
                            a += lda;
                            b += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }
        }
    }

private:
    void init(size_t M, size_t N, size_t K) {
        A_.reserve(M * K);
        B_.reserve(K * N);
        C_.reserve(M * N);
        CReference.reserve(M * N);

        utils::random_init(A_.data(), M * K);
        utils::random_init(B_.data(), K * N);
        utils::random_init(C_.data(), M * N);
        utils::random_init(CReference.data(), M * N);
    }

    void clear() {
        A_.clear();
        B_.clear();
        C_.clear();
        CReference.clear();
    }

private:
    xsdnn::mat_t A_;
    xsdnn::mat_t B_;
    xsdnn::mat_t C_;
    xsdnn::mat_t CReference;
};

int main() {
    SGemmTester tester;
    tester.ExecuteShort();
}

