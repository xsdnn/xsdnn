//
// Created by rozhin on 03.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "../xsdnn.h"
#include <gtest/gtest.h>
using namespace xsdnn;

TEST(network_cerial, seq) {
    network<sequential> net("seq_net_test");

    net << fully_connected(100, 100) << relu()
        << fully_connected(100, 100) << relu()
        << fully_connected(100, 100) << relu()
        << fully_connected(100, 100) << relu();

    net.init_weight();
    net.save("seq_net_test.xs");

    network<sequential> net2("seq_net_test");
    net2.load("seq_net_test.xs");
    ASSERT_TRUE(net == net2);
}

TEST(network_cerial, graph) {
    //TODO: дописать, когда будут слои с параметрами
}