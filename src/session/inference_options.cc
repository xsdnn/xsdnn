//
// Created by rozhin on 10.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <session/inference_options.h>

namespace xsdnn {

    std::ostream& operator<<(std::ostream& out, const InfOptions& opt) {
        out << "Inf Options: " << std::endl;
        out << "\tNumThreads : " << opt.num_threads_ << std::endl;
        out << "\tBatchSize  : " << opt.batch_size_;
        return out;
    }

} // xsdnn

