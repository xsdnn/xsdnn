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

    ASSERT_TRUE(data.size() == shp1.size());
    ASSERT_TRUE(grad.size() == shp1.size());
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

/*
 * Simple case for fullyconnected layer
 */
TEST(Node, Alloc) {
    shape3d data(1, 5, 3); // 5 features, 3 obj :: ColMajor
    shape3d w(1, 5, 10); // in = 5, out = 10
    shape3d b(1, 10, 1); // 5 features
    shape3d out(1, 10, 3); // 10 feature, 3 obj

    std::vector<tensor_type> in_type = {tensor_type::data, tensor_type::weight, tensor_type::bias};
    std::vector<tensor_type> out_type = {tensor_type::data};

    std::vector<shape3d> in_shape = {data, w, b};
    std::vector<shape3d> out_shape = {out};

    node curr_node(in_shape.size(), out_shape.size());

    // allocate input/output
    for (size_t i = 0; i < in_shape.size(); ++i) {
        curr_node.prev()[i] = std::make_shared<edge>(nullptr, in_shape[i], in_type[i]);
    }

    for (size_t i = 0; i < out_shape.size(); ++i) {
        curr_node.next()[i] = std::make_shared<edge>(&curr_node, out_shape[i], out_type[i]);
    }

    // checking concept available
    for (size_t i = 0; i < in_shape.size(); ++i) {
        Tensor_3D& _data = curr_node.prev()[i]->get_data();
        Tensor_3D& _grad = curr_node.prev()[i]->get_gradient();
        ASSERT_TRUE(_data.size() == in_shape[i].size());
        ASSERT_TRUE(_grad.size() == in_shape[i].size());
    }

    for (size_t i = 0; i < out_shape.size(); ++i) {
        Tensor_3D& _data = curr_node.next()[i]->get_data();
        Tensor_3D& _grad = curr_node.next()[i]->get_gradient();
        ASSERT_TRUE(_data.size() == out_shape[i].size());
        ASSERT_TRUE(_grad.size() == out_shape[i].size());
    }
}
#endif //XSDNN_EDGE_NODE_TEST_HPP
