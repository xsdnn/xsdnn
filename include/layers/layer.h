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
#include "../serializer/xs.proto3.pb.h"
#include <sstream>

namespace xsdnn {

class TypeHolder {
public:
    TypeHolder() {}
    TypeHolder(tensor_type T, XsDtype D) : T_(T), D_(D) {}
    tensor_type concept_type() const { return T_; }
    XsDtype     tensor_dtype() const { return D_; }

private:
    tensor_type T_;
    XsDtype D_;
};

std::vector<TypeHolder> get_typed_holder(bool has_bias, XsDtype dtype);

class layer : public node {
public:
    layer(const std::vector<TypeHolder>& in_type,
          const std::vector<TypeHolder>& out_type);
    virtual ~layer();


    void set_parallelize(bool parallelize);
    void set_num_threads(size_t num_threads);
    void set_backend(core::backend_t engine);
    core::backend_t engine() const;

    bool parallelize() const;
    size_t in_concept() const;
    size_t out_concept() const;
    bool trainable() const;

    size_t in_data_size() const;
    size_t out_data_size() const;
    std::vector<shape3d> in_data_shape() const;
    std::vector<shape3d> out_data_shape() const;

    std::vector<const tensor_t*> weights() const;
    std::vector<tensor_t*> weights();

    std::vector<edgeptr_t> inputs();
    std::vector<edgeptr_t> outputs();
    std::vector<edgeptr_t> outputs() const;

    /*
     * 1. Создается временный буффер, в который последовательно копируются
     * значения из всех тензоров весов \ смещений для данного слоя.
     *
     * 2. Временный буффер копируется в структуру для сериализации.
     *
     * 3. Сохраняются shape тензора и тип его данных для последующей десериализации.
     */
    virtual
    void save(xs::TensorInfo* dst) const {
        std::vector<const tensor_t*> Weights = weights();
        size_t WBSizeWithDtype = 1;
        for (const tensor_t* weight : Weights)
            WBSizeWithDtype *= weight->shape().size() * sizeofDtype(weight->dtype());

        std::string TmpBuffer;
        TmpBuffer.reserve(WBSizeWithDtype);
        for (const tensor_t* weight : Weights) {
            size_t TensorSizeWithDtype = weight->shape().size() * sizeofDtype(weight->dtype());
            std::string_view TensorMutableData = std::string_view(static_cast<char*>(weight->GetMutableDataRaw()), TensorSizeWithDtype);
            TmpBuffer += TensorMutableData;
        }

        std::string* MutableRawData  = dst->mutable_raw_data();
        MutableRawData->assign(TmpBuffer.data(), WBSizeWithDtype);

        for (const tensor_t* weight : Weights) {
            dst->add_dims(weight->shape().size());
        }

        XsDtype tensor0_dtype = Weights[0]->dtype();
        dst->set_type(get_xsttype_from_dtype(tensor0_dtype));
    }

    /*
     * 1. Проверка на правильность размеров по всем измерениям.
     *
     * 2. Побайтовое копирование в тензор слоя.
     */
    virtual
    void load(const xs::TensorInfo* src) {
        std::vector<tensor_t*> Weights = weights();
        assert(src->dims_size() == Weights.size());
        size_t src_size = 0, weights_size = 0;
        for (size_t d = 0; d < src->dims_size(); ++d) {
            src_size += src->dims(d);
            weights_size += Weights[d]->shape().size();
        }
        assert(src_size == weights_size);

        const std::string& DataRaw = src->raw_data();
        size_t PrevTensorSizeWithDtype = 0;
        for (tensor_t* weight : Weights) {
            size_t TensorSizeWithDtype = weight->shape().size() * sizeofDtype(weight->dtype());

            auto DataRawStartPos = DataRaw.begin() + PrevTensorSizeWithDtype;
            auto DataRawStopPos = DataRawStartPos + TensorSizeWithDtype;
            std::copy(DataRawStartPos, DataRawStopPos, static_cast<char*>(weight->GetMutableDataRaw()));
            PrevTensorSizeWithDtype += TensorSizeWithDtype;
        }

        initialized_ = true;
    }

    void set_in_data(const std::vector<BTensor>& data);
    void set_trainable(bool trainable);

    std::vector<BTensor> output() const;

    std::vector<TypeHolder> in_types() const;
    std::vector<TypeHolder> out_types() const;

    template<typename WeightInit>
    void weight_init(const WeightInit& f) {
        weight_init_ = std::make_shared<WeightInit>(f);
    }

    template<typename BiasInit>
    void bias_init(const BiasInit& f) {
        bias_init_ = std::make_shared<BiasInit>(f);
    }

    virtual
    void
    post_update() {}

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
    forward_propagation(const std::vector<BTensor*>& in_data,
                        std::vector<BTensor*>& out_data) = 0;


    void forward();

    void setup(bool reset_weight);
    void init_weight();

    virtual
    void set_sample_count(size_t sample_count);

public:
    friend void connection_mismatch(const layer& from,
                                    const layer& to);

private:
    void alloc_input(size_t i) const;
    void alloc_output(size_t i) const;
    edgeptr_t ith_in_node(size_t i);
    edgeptr_t ith_out_node(size_t i);
    tensor_t* get_weight_data(size_t i);
    const tensor_t* get_weight_data(size_t i) const;
    xs::TensorInfo::TensorType get_xsttype_from_dtype(XsDtype dtype) const;
    XsDtype get_dtype_from_xsttype(xs::TensorInfo::TensorType xsttype) const;

protected:
    bool initialized_;
    bool parallelize_;
    size_t num_threads_;
    size_t in_concept_;
    size_t out_concept_;
    std::vector<TypeHolder> in_type_;
    std::vector<TypeHolder> out_type_;

private:
    bool trainable_;
    core::backend_t engine_;
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    std::vector<BTensor *> fwd_in_data;
    std::vector<BTensor *> fwd_out_data;

    friend class GradChecker;
};

void connect(layer* last_node,
                     layer* next_node,
                     size_t last_node_data_concept_idx,
                     size_t next_node_data_concept_idx);

void connection_mismatch(const layer& from, const layer& to);

} // xsdnn

#endif //XSDNN_LAYER_H
