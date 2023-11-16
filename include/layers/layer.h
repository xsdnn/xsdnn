//
// Created by rozhin on 02.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_LAYER_H
#define XSDNN_LAYER_H

#include "../common/node.h"
#include "../utils/weight_init.h"
#include "../core/backend.h"
#include "../utils/tensor_shape.h"
#include <sstream>
#ifdef XS_USE_SERIALIZATION
#include "../serializer/xs.proto3.pb.h"
#endif

namespace xsdnn {

struct TypedHolder {
    TypedHolder() : T_(tensor_type::data), D_(xsDtype::kXsUndefined) {}
    TypedHolder(tensor_type type, xsDtype dtype) : T_(type), D_(dtype) {}

    tensor_type ttype() const noexcept { return T_; }
    xsDtype dtype() const noexcept { return D_; }

private:
    tensor_type T_;
    xsDtype D_;
};

class layer : public node {
public:
    layer(const std::vector<tensor_type>& in_type,
          const std::vector<tensor_type>& out_type,
          const xsDtype dtype);
    virtual ~layer();


    void set_parallelize(bool parallelize);
    void set_num_threads(size_t num_threads);
    void set_backend(core::backend_t engine);
    core::backend_t engine() const;

    bool parallelize() const;
    size_t in_concept() const;
    size_t out_concept() const;
    bool trainable() const;
    bool is_packed() const;

    size_t in_data_size() const;
    size_t out_data_size() const;
    std::vector<shape3d> in_data_shape() const;
    std::vector<shape3d> out_data_shape() const;

    std::vector<const mat_t*> weights() const;
    std::vector<mat_t*> weights();

    std::vector<edgeptr_t> inputs();
    std::vector<edgeptr_t> outputs();
    std::vector<edgeptr_t> outputs() const;

#ifdef XS_USE_SERIALIZATION
    virtual
    void save(xs::TensorInfo* dst) const {
        std::vector<const mat_t*> Weights = weights();
        size_t WBSizeWithDtype = 1;
        for (const mat_t* weight : Weights)
            WBSizeWithDtype *= weight->size();

        std::string TmpBuffer;
        TmpBuffer.reserve(WBSizeWithDtype);
        for (const mat_t* weight : Weights) {
            size_t TensorSizeWithDtype = weight->size();
            std::string_view TensorMutableData = std::string_view(weight->data(), TensorSizeWithDtype);
            TmpBuffer += TensorMutableData;
        }

        std::string* MutableRawData  = dst->mutable_raw_data();
        MutableRawData->assign(TmpBuffer.data(), WBSizeWithDtype);

        for (const mat_t* weight : Weights) {
            dst->add_dims(weight->size() / dtype2sizeof(this->dtype_));
        }

        dst->set_type(get_xsDtype_from_NodeDtype());
    }

    virtual
    void load(const xs::TensorInfo* src) {
        std::vector<mat_t*> Weights = weights();
        assert(src->dims_size() == Weights.size());
        size_t src_size = 0, weights_size = 0;
        for (size_t d = 0; d < src->dims_size(); ++d) {
            src_size += src->dims(d);
            weights_size += Weights[d]->size() / dtype2sizeof(this->dtype_);
        }
        assert(src_size == weights_size);

        const std::string& DataRaw = src->raw_data();
        size_t PrevTensorSizeWithDtype = 0;
        for (mat_t* weight : Weights) {
            size_t TensorSizeWithDtype = weight->size();

            auto DataRawStartPos = DataRaw.begin() + PrevTensorSizeWithDtype;
            auto DataRawStopPos = DataRawStartPos + TensorSizeWithDtype;
            std::copy(DataRawStartPos, DataRawStopPos, weight->data());
            PrevTensorSizeWithDtype += TensorSizeWithDtype;
        }

        initialized_ = true;
    }
#endif

    void set_in_data(const std::vector<tensor_t>& data);
    void set_trainable(bool trainable);

    std::vector<tensor_t> output() const;

    std::vector<tensor_type> in_types() const;
    std::vector<tensor_type> out_types() const;

    template<typename WeightInit>
    void weight_init(const WeightInit& f) {
        weight_init_ = std::make_shared<WeightInit>(f);
    }

    template<typename BiasInit>
    void bias_init(const BiasInit& f) {
        bias_init_ = std::make_shared<BiasInit>(f);
    }

    virtual
    std::vector<shape3d>
    in_shape() const = 0;

    virtual
    std::vector<shape3d>
    out_shape() const = 0;


    virtual
    std::string
    layer_type() const = 0;

    virtual
    size_t
    fan_in_size() const {
        return in_shape()[0].W;
    }

    virtual
    size_t
    fan_out_size() const {
        return out_shape()[0].W;
    }

    virtual
    void
    set_in_shape(const shape3d in_shape);


    virtual std::pair<mm_scalar, mm_scalar> out_value_range() const;

    /*
     * Forward \ backward propagation
     */

    virtual
    void
    forward_propagation(const std::vector<tensor_t*>& in_data,
                        std::vector<tensor_t*>& out_data) = 0;

    /*
     * Переупаковка концептов из одного формата данных в другой.
     * Функция вызывается на моменте конфигурирования графа,
     * и должна выполнять преобразования только над концептами weights \ bias.
     *
     * Пример:
     *      call pre_pack(chw, hwc) -> перепакует данные из формата chw в формат hwc.
     */
    virtual
    void
    pre_pack(xsMemoryFormat from, xsMemoryFormat to);

    /*
     * Переупаковка концептов из одного формата данных в другой.
     * Функция вызывается в момент исполнения графа, и должна выполнять
     * преобразования только над концептами data.
     *
     * Пример:
     *      call pack(hwc, chw) -> перепакует данные из формата hwc в формат chw.
     */
    virtual
    void
    pack(xsMemoryFormat from, xsMemoryFormat to);

    void forward();

    void setup(bool reset_weight);

    void init_weight();

    virtual
    void set_sample_count(size_t sample_count);

public:
    friend void connection_mismatch(const layer& from,
                                    const layer& to);

protected:
    void alloc_input(size_t i) const;
    void alloc_output(size_t i) const;
    edgeptr_t ith_in_node(size_t i);
    edgeptr_t ith_out_node(size_t i);
    mat_t* get_weight_data(size_t i);
    const mat_t* get_weight_data(size_t i) const;

#ifdef XS_USE_SERIALIZATION
    xs::TensorInfo::TensorType get_xsDtype_from_NodeDtype() const;
#endif

protected:
    bool initialized_;
    bool parallelize_;
    bool is_packed_{false};
    size_t num_threads_;
    size_t in_concept_;
    size_t out_concept_;
    std::vector<tensor_type> in_type_;
    std::vector<tensor_type> out_type_;

private:
    core::backend_t engine_;
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    std::vector<tensor_t*> fwd_in_data;
    std::vector<tensor_t*> fwd_out_data;
};

void connect(layer* last_node,
                     layer* next_node,
                     size_t last_node_data_concept_idx,
                     size_t next_node_data_concept_idx);

void connection_mismatch(const layer& from, const layer& to);

} // xsdnn

#endif //XSDNN_LAYER_H
