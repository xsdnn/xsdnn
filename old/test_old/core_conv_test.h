//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//


#ifndef XSDNN_CORE_CONV_TEST_H
#define XSDNN_CORE_CONV_TEST_H

TEST(conv_core, choice_engine) {
    core::conv2d engine = core::conv2d::mec;
    std::cout << engine << std::endl;

    engine = core::conv2d::im2col;
    std::cout << engine << std::endl;

    engine = core::conv2d::fft;
    std::cout << engine << std::endl;
}

#endif //XSDNN_CORE_CONV_TEST_H
