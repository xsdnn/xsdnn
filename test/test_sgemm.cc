//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <iostream>
using namespace mmpack;

#define M 5
#define N 6
#define K 4

TEST(sgemm, NoTrans_NoTrans) {
    xsdnn::mat_t A; A.reserve(M * K);
    xsdnn::mat_t B; B.reserve(K * N);
    xsdnn::mat_t C; C.reserve(M * N);

    utils::init(A.data(), M, K);
    utils::init(B.data(), K, N);


    MmGemm(
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            M, N, K, 1.0,
            A.data(), K,
            B.data(), N,
            0.0,
            C.data(), N);

    mm_scalar ExpectedArr[] {84, 90, 96, 102, 108, 114,
                             228, 250, 272, 294, 316, 338,
                             372,  410,  448,  486,  524,  562,
                             516,  570,  624,  678,  732,  786,
                             660,  730,  800,  870,  940, 1010};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ASSERT_EQ(C[i * N + j], ExpectedArr[i * N + j]);
        }
    }
}

TEST(sgemm, NoTrans_Trans) {
    xsdnn::mat_t A; A.reserve(M * K);
    xsdnn::mat_t B; B.reserve(K * N);
    xsdnn::mat_t C; C.reserve(M * N);

    utils::init(A.data(), M, K);
    utils::init(B.data(), N, K);

    MmGemm(
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            M, N, K, 1.0,
            A.data(), K,
            B.data(), K,
            0.0,
            C.data(), N);

    mm_scalar ExpectedArr[] {14, 38, 62, 86, 110, 134,
                             38, 126,  214,  302,  390,  478,
                             62,  214,  366,  518,  670,  822,
                             86,  302,  518,  734,  950, 1166,
                             110,  390,  670,  950, 1230, 1510};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ASSERT_EQ(C[i * N + j], ExpectedArr[i * N + j]);
        }
    }
}

TEST(sgemm, Trans_NoTrans) {
    xsdnn::mat_t A; A.reserve(M * K);
    xsdnn::mat_t B; B.reserve(K * N);
    xsdnn::mat_t C; C.reserve(M * N);

    utils::init(A.data(), K, M);
    utils::init(B.data(), K, N);


    MmGemm(
            CBLAS_TRANSPOSE::CblasTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            M, N, K, 1.0,
            A.data(), M,
            B.data(), N,
            0.0,
            C.data(), N);

    mm_scalar ExpectedArr[] {420, 450, 480, 510, 540, 570,
                             456, 490, 524, 558, 592, 626,
                             492, 530, 568, 606, 644, 682,
                             528, 570, 612, 654, 696, 738,
                             564, 610, 656, 702, 748, 794};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ASSERT_EQ(C[i * N + j], ExpectedArr[i * N + j]);
        }
    }
}

TEST(sgemm, Trans_Trans) {
    xsdnn::mat_t A; A.reserve(M * K);
    xsdnn::mat_t B; B.reserve(K * N);
    xsdnn::mat_t C; C.reserve(M * N);

    utils::init(A.data(), K, M);
    utils::init(B.data(), N, K);

    MmGemm(
            CBLAS_TRANSPOSE::CblasTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            M, N, K, 1.0,
            A.data(), M,
            B.data(), K,
            0.0,
            C.data(), N);

    mm_scalar ExpectedArr[] {70, 190, 310, 430, 550, 670,
                             76, 212, 348, 484, 620, 756,
                             82, 234, 386, 538, 690, 842,
                             88, 256, 424, 592, 760, 928,
                             94, 278, 462, 646, 830, 1014};

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ASSERT_EQ(C[i * N + j], ExpectedArr[i * N + j]);
        }
    }
}