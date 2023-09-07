//
// Created by rozhin on 06.09.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <cstdlib>
#include <vector>
#include "../include/utils/broadcaster.h"

int main() {
    xsdnn::shape3d shape1(1, 8, 1);
    xsdnn::shape3d shape2(1, 1, 9);

    xsdnn::broadcaster br(shape2, shape1);
    std::cout << br.get_span_size();
}

