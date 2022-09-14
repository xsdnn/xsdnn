//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_INCLUDE_CRLAYER_H
#define XSDNN_INCLUDE_CRLAYER_H

# include <map>
# include <string>
# include <stdexcept>

# include "../Config.h"

# include "../Utils/Enum.h"

# include "../Layer.h"
# include "../Layer/FullyConnected.h"
# include "../Layer/Dropout.h"
# include "../Layer/BatchNormalization.h"

# include "../Activation/ReLU.h"
# include "../Activation/LeakyReLU.h"
# include "../Activation/Sigmoid.h"
# include "../Activation/Softmax.h"
# include "../Activation/Identity.h"

# include "../Distribution/Uniform.h"
# include "../Distribution/Normal.h"
# include "../Distribution/Exponential.h"
# include "../Distribution/Constant.h"


# include "../Output.h"
# include "../Output/BinaryClassEntropy.h"
# include "../Output/MultiClassEntropy.h"
# include "../Output/RegressionMSE.h"


namespace internal {
    /// Автоматическое создание слоя из Meta файла сетки.
    /// \param map словарь с информацией о сети
    /// \param index индекс слоя
    /// \return указатель на созданный и _инициализированный_ слой
    inline Layer *create_layer(const std::map<std::string, Scalar> &map, const int &index) {
        std::string ind = std::to_string(index);
        const int layer_id = static_cast<int>(map.find("Layer " + ind)->second);
        const int activation_id = static_cast<int>(map.find("Activation " + ind)->second);
        const int distribution_id = static_cast<int>(map.find("Distribution " + ind)->second);

        Layer *layer;

        if (layer_id == FULLYCONNECTED) {
            const int in_size = static_cast<int>(map.find("in_size " + ind)->second);
            const int out_size = static_cast<int>(map.find("out_size " + ind)->second);
            const int bias = static_cast<int>(map.find("Bias FC " + ind)->second);

            switch (activation_id) {
                case IDENTITY:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<init::Uniform, activate::Identity>(in_size, out_size, bias);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<init::Exponential, activate::Identity>(in_size, out_size, bias);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<init::Normal, activate::Identity>(in_size, out_size, bias);
                            break;

                        case CONSTANT:
                            layer = new FullyConnected<init::Constant, activate::Identity>(in_size, out_size, bias);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case LEAKYRELU:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<init::Uniform, activate::LeakyReLU>(in_size, out_size, bias);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<init::Exponential, activate::LeakyReLU>(in_size, out_size, bias);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<init::Normal, activate::LeakyReLU>(in_size, out_size, bias);
                            break;

                        case CONSTANT:
                            layer = new FullyConnected<init::Constant, activate::LeakyReLU>(in_size, out_size, bias);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case RELU:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<init::Uniform, activate::ReLU>(in_size, out_size, bias);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<init::Exponential, activate::ReLU>(in_size, out_size, bias);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<init::Normal, activate::ReLU>(in_size, out_size, bias);
                            break;

                        case CONSTANT:
                            layer = new FullyConnected<init::Constant, activate::ReLU>(in_size, out_size, bias);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case SIGMOID:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<init::Uniform, activate::Sigmoid>(in_size, out_size, bias);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<init::Exponential, activate::Sigmoid>(in_size, out_size, bias);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<init::Normal, activate::Sigmoid>(in_size, out_size, bias);
                            break;

                        case CONSTANT:
                            layer = new FullyConnected<init::Constant, activate::Sigmoid>(in_size, out_size, bias);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case SOFTMAX:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<init::Uniform, activate::Softmax>(in_size, out_size, bias);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<init::Exponential, activate::Softmax>(in_size, out_size, bias);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<init::Normal, activate::Softmax>(in_size, out_size, bias);
                            break;

                        case CONSTANT:
                            layer = new FullyConnected<init::Constant, activate::Softmax>(in_size, out_size, bias);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;

                default:
                    throw std::invalid_argument("[function create_layer]: Activation is not of a known type");

            }
        } else if (layer_id == DROPOUT) {
            const int in_size = static_cast<int>(map.find("in_size " + ind)->second);
            const Scalar drop_rate = map.find("dropout_rate " + ind)->second;

            switch (activation_id) {
                case IDENTITY:
                    layer = new Dropout<activate::Identity>(in_size, drop_rate);
                    break;

                case LEAKYRELU:
                    layer = new Dropout<activate::LeakyReLU>(in_size, drop_rate);
                    break;

                case RELU:
                    layer = new Dropout<activate::ReLU>(in_size, drop_rate);
                    break;

                case SIGMOID:
                    layer = new Dropout<activate::Sigmoid>(in_size, drop_rate);
                    break;

                case SOFTMAX:
                    layer = new Dropout<activate::Softmax>(in_size, drop_rate);
                    break;

                default:
                    throw std::invalid_argument("[function create_layer]: Activation is not of a known type");
            }
        } else if (layer_id == BATCHNORM1D) {
            const int affine = static_cast<int>(map.find("Affine " + ind)->second);
            const int in_size = static_cast<int>(map.find("in_size " + ind)->second);
            const Scalar eps = static_cast<Scalar>(map.find("tolerance " + ind)->second);
            const Scalar momentum = static_cast<Scalar>(map.find("momentum " + ind)->second);


            switch (activation_id) {
                case IDENTITY:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new BatchNorm1D<init::Uniform, activate::Identity>(in_size, affine, eps, momentum);
                            break;

                        case EXPONENTIAL:
                            layer = new BatchNorm1D<init::Exponential, activate::Identity>(in_size, affine, eps,
                                                                                           momentum);
                            break;

                        case NORMAL:
                            layer = new BatchNorm1D<init::Normal, activate::Identity>(in_size, affine, eps, momentum);
                            break;

                        case CONSTANT:
                            layer = new BatchNorm1D<init::Constant, activate::Identity>(in_size, affine, eps, momentum);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case LEAKYRELU:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new BatchNorm1D<init::Uniform, activate::LeakyReLU>(in_size, affine, eps, momentum);
                            break;

                        case EXPONENTIAL:
                            layer = new BatchNorm1D<init::Exponential, activate::LeakyReLU>(in_size, affine, eps,
                                                                                            momentum);
                            break;

                        case NORMAL:
                            layer = new BatchNorm1D<init::Normal, activate::LeakyReLU>(in_size, affine, eps, momentum);
                            break;

                        case CONSTANT:
                            layer = new BatchNorm1D<init::Constant, activate::LeakyReLU>(in_size, affine, eps,
                                                                                         momentum);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case RELU:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new BatchNorm1D<init::Uniform, activate::ReLU>(in_size, affine, eps, momentum);
                            break;

                        case EXPONENTIAL:
                            layer = new BatchNorm1D<init::Exponential, activate::ReLU>(in_size, affine, eps, momentum);
                            break;

                        case NORMAL:
                            layer = new BatchNorm1D<init::Normal, activate::ReLU>(in_size, affine, eps, momentum);
                            break;

                        case CONSTANT:
                            layer = new BatchNorm1D<init::Constant, activate::ReLU>(in_size, affine, eps, momentum);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case SIGMOID:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new BatchNorm1D<init::Uniform, activate::Sigmoid>(in_size, affine, eps, momentum);
                            break;

                        case EXPONENTIAL:
                            layer = new BatchNorm1D<init::Exponential, activate::Sigmoid>(in_size, affine, eps,
                                                                                          momentum);
                            break;

                        case NORMAL:
                            layer = new BatchNorm1D<init::Normal, activate::Sigmoid>(in_size, affine, eps, momentum);
                            break;

                        case CONSTANT:
                            layer = new BatchNorm1D<init::Constant, activate::Sigmoid>(in_size, affine, eps, momentum);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case SOFTMAX:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new BatchNorm1D<init::Uniform, activate::Softmax>(in_size, affine, eps, momentum);
                            break;

                        case EXPONENTIAL:
                            layer = new BatchNorm1D<init::Exponential, activate::Softmax>(in_size, affine, eps,
                                                                                          momentum);
                            break;

                        case NORMAL:
                            layer = new BatchNorm1D<init::Normal, activate::Softmax>(in_size, affine, eps, momentum);
                            break;

                        case CONSTANT:
                            layer = new BatchNorm1D<init::Constant, activate::Softmax>(in_size, affine, eps, momentum);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;

                default:
                    throw std::invalid_argument("[function create_layer]: Activation is not of a known type");

            }
        } else {
            throw std::invalid_argument("[function create_layer]: Layer is not of a known type");
        }

        layer->init();
        return layer;
    }

    /// Автоматическая установка выходного слоя из Meta файла сетки
    /// \param map словарь с информацией о сети.
    /// \return указатель на выходной слой
    inline Output *create_output(const std::map<std::string, Scalar> &map) {
        const int output_id = static_cast<int>(map.find("OutputLayer")->second);
        Output *output;

        switch (output_id) {
            case REGRESSIONMSE:
                output = new MSELoss();
                break;

            case BINARYCLASSENTROPY:
                output = new BinaryEntropyLoss();
                break;

            case MULTICLASSENTROPY:
                output = new CrossEntropyLoss();
                break;

            default:
                throw std::invalid_argument("[function create_output]: Output is not of a known type");
        }

        return output;
    }
}

#endif //XSDNN_INCLUDE_CRLAYER_H
