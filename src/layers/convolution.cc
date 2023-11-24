//
// Created by rozhin on 21.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include <layers/convolution.h>
#include <utils/transpose.h>

namespace xsdnn {

void conv::set_params(size_t in_channel, size_t in_height, size_t in_width, size_t out_channel, size_t group_count, bool has_bias,
                      std::vector<size_t> kernel_shape,
                      std::vector<size_t> stride_shape,
                      std::vector<size_t> dilation_shape,
                      padding_mode pad_type,
                      std::vector<size_t> pads,
                      MmActivationType activation_type) {
    if (is_1D_tensor(shape3d(in_channel, in_height, in_width))) {
        params_._.Dimensions = 1;
    } else if (is_2D_tensor(shape3d(in_channel, in_height, in_width))) {
        params_._.Dimensions = 2;
    } else {
        xs_error("Unsupported dimensions in input data of conv layer");
    }
    params_.infer_output_requirement_shape(shape3d(in_channel, in_height, in_width),
                                           out_channel, group_count, has_bias,
                                           kernel_shape, stride_shape,
                                           dilation_shape, pad_type, pads, activation_type);
}

void conv::configure(core::backend_t engine) {
    if (engine == core::backend_t::xnnpack)
#ifdef XS_USE_XNNPACK
        fwd_kernel_->CreateAndReshapeXNNKernel(this->dtype(), weights(), params_);
#else
        throw xs_error(START_MSG + "This build doesn't support XNN Backend Engine. "
                                   "Rebuild with -Dxsdnn_BUILD_XNNPACK_ENGINE=ON");
#endif
}

void conv::init_backend(core::backend_t engine) {
    fwd_kernel_.reset(new core::ConvFwdKernel);
    set_backend(engine);
}

std::vector<shape3d> conv::in_shape() const {
    if (params_._.Dimensions == 2) {
        size_t in_channel = params_._.InChannel * params_._.GroupCount;
        size_t in_height = params_._.InShape[0];
        size_t in_width = params_._.InShape[1];
        size_t f_count = params_._.FilterCount;
        size_t out_channel = f_count * params_._.GroupCount;
        if (params_._.Bias) {
            return {
                    shape3d(in_channel, in_height, in_width),
                    shape3d(out_channel * params_._.InChannel, params_._.KernelShape[0], params_._.KernelShape[1]),
                    shape3d(out_channel, 1, 1)
            };
        } else {
            return {
                    shape3d(in_channel, in_height, in_width),
                    shape3d(out_channel * params_._.InChannel, params_._.KernelShape[0], params_._.KernelShape[1]),
            };
        }
    } else {
        throw xs_error("conv in_shape for 1D not implemented yet");
    }

}

std::vector<shape3d> conv::out_shape() const {
    if (params_._.Dimensions == 2) {
        size_t out_channel = params_._.FilterCount * params_._.GroupCount;
        return { shape3d(out_channel, params_._.OutShape[0], params_._.OutShape[1]) };
    } else {
        throw xs_error("conv out_shape for 1D not implemented yet");
    }
}

params::conv conv::get_params() const {
    return params_;
}

std::string conv::layer_type() const {
    return "conv";
}

void conv::forward_propagation(const std::vector<tensor_t *> &in_data,
                                   std::vector<tensor_t *> &out_data) {
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.set_parallelize(this->parallelize());
    fwd_ctx_.set_engine(this->engine());
    fwd_ctx_.set_num_threads(this->num_threads_);
    fwd_ctx_.set_dtype(layer::dtype());

    fwd_kernel_->Compute(fwd_ctx_, params_);
}

void conv::pre_pack(xsdnn::xsMemoryFormat from, xsdnn::xsMemoryFormat to) {
    if (from == to) return;
    if (from == xsMemoryFormat::chw & to == xsMemoryFormat::hwc) {
        if (params_._.Dimensions != 2) throw xs_error(START_MSG + "Pre packing weights not implemented for 1D Conv");
        size_t out_channel = params_._.FilterCount * params_._.GroupCount;

        std::vector<size_t> WShapeSpan = {out_channel, params_._.InChannel, params_._.KernelShape[0], params_._.KernelShape[1]};

        mat_t* Weights = this->get_weight_data(1);
        mat_t PackedWeights(Weights->size());

        xs_single_axis_transpose(Weights, WShapeSpan,
                                 &PackedWeights, this->dtype(), 1, 3);
        *Weights = PackedWeights;

        is_packed_ = true;
    } else {
        throw xs_error(START_MSG + "Pre packing weights for different from chw->hwc computation not implemented yet");
    }
}

void conv::pack(xsdnn::xsMemoryFormat from, xsdnn::xsMemoryFormat to) {
    if (from == to) return;
    if (from == xsMemoryFormat::hwc && to == xsMemoryFormat::chw) {
        // todo: impl this
    }
}

} // xsdnn
