//
// Created by shuffle on 27.06.22.
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

# include "../Activation/ReLU.h"
# include "../Activation/Sigmoid.h"
# include "../Activation/Softmax.h"
# include "../Activation/Identity.h"

# include "../Distribution/Uniform.h"
# include "../Distribution/Normal.h"
# include "../Distribution/Exponential.h"


# include "../Output.h"
# include "../Output/BinaryClassEntropy.h"
# include "../Output/MultiClassEntropy.h"
# include "../Output/RegressionMSE.h"


namespace internal
{
    /// Автоматическое создание слоя из Meta файла сетки.
    /// \param map словарь с информацией о сети
    /// \param index индекс слоя
    /// \return указатель на созданный и _инициализированный_ слой
    inline Layer* create_layer(const std::map<std::string, int>& map, const int& index)
    {
        std::string ind = std::to_string(index);
        const int layer_id = map.find("Layer " + ind)->second;
        const int activation_id = map.find("Activation " + ind)->second;
        const int distribution_id = map.find("Distribution " + ind)->second;

        Layer* layer;

        if (layer_id == FULLYCONNECTED)
        {
            const int in_size = map.find("in_size " + ind)->second;
            const int out_size = map.find("out_size " + ind)->second;

            switch (activation_id) {
                case IDENTITY:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<Identity, Uniform>(in_size, out_size);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<Identity, Exponential>(in_size, out_size);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<Identity, Normal>(in_size, out_size);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;



                case RELU:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<ReLU, Uniform>(in_size, out_size);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<ReLU, Exponential>(in_size, out_size);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<ReLU, Normal>(in_size, out_size);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;



                case SIGMOID:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<Sigmoid, Uniform>(in_size, out_size);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<Sigmoid, Exponential>(in_size, out_size);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<Sigmoid, Normal>(in_size, out_size);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;


                case SOFTMAX:

                    switch (distribution_id) {
                        case UNIFORM:
                            layer = new FullyConnected<Softmax, Uniform>(in_size, out_size);
                            break;

                        case EXPONENTIAL:
                            layer = new FullyConnected<Softmax, Exponential>(in_size, out_size);
                            break;

                        case NORMAL:
                            layer = new FullyConnected<Softmax, Normal>(in_size, out_size);
                            break;

                        default:
                            throw std::invalid_argument("[function create_layer]: Distribution is not of a known type");
                    }
                    break;

                default:
                    throw std::invalid_argument("[function create_layer]: Activation is not of a known type");

            }
        } else
        {
            throw std::invalid_argument("[function create_layer]: Layer is not of a known type");
        }

        layer->init();
        return layer;
    }

    /// Автоматическая установка выходного слоя из Meta файла сетки
    /// \param map словарь с информацией о сети.
    /// \return указатель на выходной слой
    inline Output* create_output(const std::map<std::string, int>& map)
    {
        const int output_id = map.find("OutputLayer")->second;
        Output* output;

        switch (output_id) {
            case REGRESSIONMSE:
                output = new RegressionMSE();
                break;

            case BINARYCLASSENTROPY:
                output = new BinaryClassEntropy();
                break;

            default:
                throw std::invalid_argument("[function create_output]: Output is not of a known type");
        }

        return output;
    }
}

#endif //XSDNN_INCLUDE_CRLAYER_H
