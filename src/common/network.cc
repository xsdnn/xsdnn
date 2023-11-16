//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <common/network.h>
#include <core/framework/threading.h>
#include <fstream>

namespace xsdnn {

void network::init_weight() {
    net_.setup(true);
}

void network::set_num_threads(size_t num_threads) noexcept {
    net_.user_num_threads_ = num_threads;
}

bool network::empty() const {
    return net_.nodes_.empty();
}

mat_t network::predict(const mat_t &in) {
    fprop(in);
    assert(net_.output_layers_.size() == 1);
    return {};
}

tensor_t network::predict(const tensor_t &in) {
    fprop(in);
}


void network::fprop(const mat_t &in) {
    if (!configured_) throw xs_error(START_MSG + "Network not configured. Run network().configure()");
    net_.forward(in);
}

void network::fprop(const tensor_t &in) {
    if (!configured_) throw xs_error(START_MSG + "Network not configured. Run network().configure()");
    net_.forward(in);
}


void network::save(const std::string filename) {
    net_.save_model(filename, network_name_);
}

void network::load(const std::string filename) {
    net_.load_model(filename);
}

void construct_graph(network& net,
                     const std::vector<layer*>& input,
                     const std::vector<layer*>& out) {
    net.net_.construct(input, out);
}

void network::configure() {
    if (configured_) return;

    // Configure memory format for each node
    net_.mtypes_.resize(net_.nodes_.size(), xsMemoryFormat::chw);

    if (net_.have_engine_xnnpack()) {
#ifdef XS_USE_XNNPACK
        core::XNNCompiler::getInstance().initialize();
        concurrency::threadpool::getInstance().create(net_.user_num_threads_);
        if (net_.get_num_xnnpack_backend_engine() != net_.nodes_.size()) {
            throw xs_error(START_MSG + "Attention! This graph is executed on different Backend Engines.\n"
                                   "It is recommended to use only one Backend Engine for all types of nodes"
                                   " to avoid the overhead of data repackaging.");
        }
        xs_warning(START_MSG + "This Graph has XNNPACK Backend Engine node's."
                               "\nMake sure that the input data is in the NHWC memory format.");

        for (size_t i = 0; i < net_.nodes_.size(); ++i) {
            if (net_.nodes_[i]->engine() == core::backend_t::xnnpack) {
                net_.mtypes_[i] = xsMemoryFormat::hwc;
                if (!net_.nodes_[i]->is_packed())
                    // from -> to
                    net_.nodes_[i]->pre_pack(xsMemoryFormat::chw, xsMemoryFormat::hwc);
            }
        }
#else
        throw xs_error("This xsdnn build doesn't support XNNPACK backend engine. Recompile with xsdnn_BUILD_XNNPACK_ENGINE=ON");
#endif
    }

    configured_ = true;
}

} // xsdnn


