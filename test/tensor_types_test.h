//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//


#ifndef XSDNN_TENSOR_TYPES_TEST_H
#define XSDNN_TENSOR_TYPES_TEST_H

TEST(tensor, parallel) {
    internal::display::Timer t;
//    xsTypes::Matrix T_1000x1000(1000, 1000); T_1000x1000.setRandom();
//    xsTypes::Matrix T_1000x1000_2(1000, 1000); T_1000x1000_2.setRandom();
//    xsTypes::Matrix Res(1000, 1000);
//    // Create the Eigen ThreadPool
//    Eigen::ThreadPool pool(8 /* number of threads in pool */);
//    // Create the Eigen ThreadPoolDevice.
//    Eigen::ThreadPoolDevice my_device(&pool, pool.NumThreads() /* number of threads to use */);
//    t.start();
//    for (int i = 0; i < 1500; i++) {
//        Res.device(my_device) = T_1000x1000 * T_1000x1000_2;
//    }
//    std::cout << "Eigen::Tensor " << t.elapced() << " sec" << std::endl;

    Matrix m_1(1000, 1000); m_1.setRandom();
    Matrix m_2(1000, 1000); m_2.setRandom();
    Matrix res(1000, 1000); res.setRandom();

    t.restart();
    for (int i = 0; i < 1500; i++) {
        res.array() = m_1.array() * m_2.array();
    }
    std::cout << "Eigen::Matrix " << t.elapced() << " sec" << std::endl;
////    std::cout << sysconf(_SC_NPROCESSORS_ONLN); // num available threads
//    t.restart();
//    Eigen::TensorMap<Eigen::Tensor<Scalar, 2>, 2> M_1(m_1.data(), 1000, 1000);
//    Eigen::TensorMap<Eigen::Tensor<Scalar, 2>, 2> M_2(m_2.data(), 1000, 1000);
//    Eigen::TensorMap<Eigen::Tensor<Scalar, 2>, 2> Res_(res.data(), 1000, 1000);
//    for (int i = 0; i < 1500; i++) {
//        Res_.device(my_device) = M_1 * M_2;
//    }
//    std::cout << "Eigen::TensorMap " << t.elapced() << " sec" << std::endl;
}



#endif //XSDNN_TENSOR_TYPES_TEST_H
