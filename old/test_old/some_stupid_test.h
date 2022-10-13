//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//


#ifndef XSDNN_SOME_STUPID_TEST_H
#define XSDNN_SOME_STUPID_TEST_H

TEST(tensor, type) {
    xsTypes<Scalar>::Matrix m(2, 4);
    m.setRandom();
    std::cout << "Matrix -> \n" << m << "\n\n\n\n";
}

#endif //XSDNN_SOME_STUPID_TEST_H
