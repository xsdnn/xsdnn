//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "xsdnn.h"
#include "test_utils.h"
#include <iostream>
using namespace mmpack;

#define M 5
#define N 6
#define K 4

TEST(sgemm, NoTrans_NoTrans) {
    xsdnn::mat_t A(M * K * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t B(K * N * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t C(M * N * dtype2sizeof(xsdnn::kXsFloat32));

    utils::sequantial_init_fp32(A, M, K);
    utils::sequantial_init_fp32(B, K, N);

    gsl::span<const float> ASpan = GetDataAsSpan<const float>(&A);
    gsl::span<const float> BSpan = GetDataAsSpan<const float>(&B);
    gsl::span<float> CSpan = GetMutableDataAsSpan<float>(&C);

    MmGemm(
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            M, N, K, 1.0,
            ASpan.data(), K,
            BSpan.data(), N,
            0.0,
            CSpan.data(), N);

    float ExpectedArr[] {84, 90, 96, 102, 108, 114,
                             228, 250, 272, 294, 316, 338,
                             372,  410,  448,  486,  524,  562,
                             516,  570,  624,  678,  732,  786,
                             660,  730,  800,  870,  940, 1010};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            utils::xsAssert_eq(CSpan[i * N + j], ExpectedArr[i * N + j], kXsFloat32);
        }
    }
}

TEST(sgemm, NoTrans_Trans) {
    xsdnn::mat_t A(M * K * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t B(K * N * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t C(M * N * dtype2sizeof(xsdnn::kXsFloat32));

    utils::sequantial_init_fp32(A, M, K);
    utils::sequantial_init_fp32(B, N, K);

    gsl::span<const float> ASpan = GetDataAsSpan<const float>(&A);
    gsl::span<const float> BSpan = GetDataAsSpan<const float>(&B);
    gsl::span<float> CSpan = GetMutableDataAsSpan<float>(&C);

    MmGemm(
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            M, N, K, 1.0,
            ASpan.data(), K,
            BSpan.data(), K,
            0.0,
            CSpan.data(), N);

    mm_scalar ExpectedArr[] {14, 38, 62, 86, 110, 134,
                             38, 126,  214,  302,  390,  478,
                             62,  214,  366,  518,  670,  822,
                             86,  302,  518,  734,  950, 1166,
                             110,  390,  670,  950, 1230, 1510};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            utils::xsAssert_eq(CSpan[i * N + j], ExpectedArr[i * N + j], kXsFloat32);
        }
    }
}

TEST(sgemm, Trans_NoTrans) {
    xsdnn::mat_t A(M * K * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t B(K * N * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t C(M * N * dtype2sizeof(xsdnn::kXsFloat32));

    utils::sequantial_init_fp32(A, K, M);
    utils::sequantial_init_fp32(B, K, N);

    gsl::span<const float> ASpan = GetDataAsSpan<const float>(&A);
    gsl::span<const float> BSpan = GetDataAsSpan<const float>(&B);
    gsl::span<float> CSpan = GetMutableDataAsSpan<float>(&C);

    MmGemm(
            CBLAS_TRANSPOSE::CblasTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            M, N, K, 1.0,
            ASpan.data(), M,
            BSpan.data(), N,
            0.0,
            CSpan.data(), N);

    mm_scalar ExpectedArr[] {420, 450, 480, 510, 540, 570,
                             456, 490, 524, 558, 592, 626,
                             492, 530, 568, 606, 644, 682,
                             528, 570, 612, 654, 696, 738,
                             564, 610, 656, 702, 748, 794};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            utils::xsAssert_eq(CSpan[i * N + j], ExpectedArr[i * N + j], kXsFloat32);
        }
    }
}

TEST(sgemm, Trans_Trans) {
    xsdnn::mat_t A(M * K * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t B(K * N * dtype2sizeof(xsdnn::kXsFloat32));
    xsdnn::mat_t C(M * N * dtype2sizeof(xsdnn::kXsFloat32));

    utils::sequantial_init_fp32(A, K, M);
    utils::sequantial_init_fp32(B, N, K);

    gsl::span<const float> ASpan = GetDataAsSpan<const float>(&A);
    gsl::span<const float> BSpan = GetDataAsSpan<const float>(&B);
    gsl::span<float> CSpan = GetMutableDataAsSpan<float>(&C);

    MmGemm(
            CBLAS_TRANSPOSE::CblasTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            M, N, K, 1.0,
            ASpan.data(), M,
            BSpan.data(), K,
            0.0,
            CSpan.data(), N);

    mm_scalar ExpectedArr[] {70, 190, 310, 430, 550, 670,
                             76, 212, 348, 484, 620, 756,
                             82, 234, 386, 538, 690, 842,
                             88, 256, 424, 592, 760, 928,
                             94, 278, 462, 646, 830, 1014};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            utils::xsAssert_eq(CSpan[i * N + j], ExpectedArr[i * N + j], kXsFloat32);
        }
    }
}