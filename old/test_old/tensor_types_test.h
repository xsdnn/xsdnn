//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//


#ifndef XSDNN_TENSOR_TYPES_TEST_H
#define XSDNN_TENSOR_TYPES_TEST_H

TEST(tensor, parallel) {
    internal::display::Timer t;
    xsTypes::Matrix T_1000x1000(4094, 4094); T_1000x1000.setRandom();
    xsTypes::Matrix T_1000x1000_2(4094, 4094); T_1000x1000_2.setRandom();
    xsTypes::Matrix Res(4094, 4094);
    t.start();
    for (int i = 0; i < 1500; i++) {
        Res.device(xsThread::CPUDevice) = T_1000x1000 * T_1000x1000_2;
    }
    T_1000x1000.chip(2, 0).maximum();
    std::cout << "Eigen::Tensor " << t.elapced() << " sec" << std::endl;
    xsTypes::AlignedMapMatrix AT_1000x1000(T_1000x1000.data(), 4094, 4094);
    xsTypes::AlignedMapMatrix AT_1000x1000_2(T_1000x1000_2.data(), 4094, 4094);
    xsTypes::AlignedMapMatrix ARes(Res.data(), 4094, 4094);
    t.restart();
    for (int i = 0; i < 1500; i++) {
        ARes.device(xsThread::CPUDevice) = AT_1000x1000 * AT_1000x1000_2;
    }
    std::cout << "Eigen::TensorMapAligned " << t.elapced() << " sec" << std::endl;
}



#endif //XSDNN_TENSOR_TYPES_TEST_H
