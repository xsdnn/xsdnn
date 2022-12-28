//
// Created by Andrei R. on 29.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_EDGE_NODE_TEST_HPP
#define XSDNN_EDGE_NODE_TEST_HPP

TEST(Edge, Alloc) {
    shape3d shp1(3, 4, 4);
    edge e1(nullptr, shp1, tensor_type::data);

    ASSERT_TRUE(e1.prev() == nullptr);
    ASSERT_TRUE(e1.ttype() == tensor_type::data);
    ASSERT_TRUE(e1.next().size() == 0);

    Tensor_3D& data = e1.get_data();
    Tensor_3D& grad = e1.get_gradient();

    ASSERT_TRUE(data.size() == 48);
    ASSERT_TRUE(grad.size() == 48);
}

TEST(Edge, MakeDataInit) {
    shape3d shp1(3, 2, 2);
    edge e1(nullptr, shp1, tensor_type::data);

    Tensor_3D& data = e1.get_data();
    data.setZero();

    std::ostringstream os;
    os << data;
    EXPECT_EQ(os.str(), "0 0\n0 0\n\n0 0\n0 0\n\n0 0\n0 0");
}

// TODO: Add test for Node

#endif //XSDNN_EDGE_NODE_TEST_HPP
