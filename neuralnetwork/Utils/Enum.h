#pragma once

///
/// Этот файл служит как вспомогательная утилита для определения типа слоя, его функции активации и выходного слоя сетки.
/// Все представлено ввиде enum перечисления, что позволит масштабировать эту систему до нужных размеров.
/// 

# include <stdexcept>

namespace internal
{
    /// ID слоя
	enum LAYER_TYPE_ENUM
	{
		FULLYCONNECTED = 0 ///< Полносвязный
	};

	///
	/// \param type тип слоя
	/// \return ID слоя
	inline int layer_id(const std::string& type)
	{
		if (type == "FullyConnected") return FULLYCONNECTED;

		throw std::invalid_argument("[function layer_id]: unknown type of layer");
	}

    /// ID функции активации
	enum ACTIVATION_FUNC_ENUM
	{
		RELU = 0,               ///< ReLU
		SIGMOID,                ///< Sigmoid
        IDENTITY,               ///< Identity
        SOFTMAX                 ///< Softmax
	};

	///
	/// \param type тип функции активации
	/// \return ID функции активации
	inline int activation_id(const std::string& type)
	{
		if (type == "ReLU") return RELU;
		if (type == "Sigmoid") return SIGMOID;
        if (type == "Identity") return IDENTITY;
        if (type == "Softmax") return SOFTMAX;
        
		throw std::invalid_argument("[function activation_id]: unknown type of activation func");
	}

    /// ID выходного слоя
	enum OUTPUT_ENUM
	{
		REGRESSIONMSE = 0,              ///< Регрессия - критерий MSE
        BINARYCLASSENTROPY              ///< Бинарная классификация
	};

	///
	/// \param type тип выходного слоя
	/// \return ID выходного слоя
	inline int output_id(const std::string& type)
	{
		if (type == "RegressionMSE") return REGRESSIONMSE;
        if (type == "BinaryClassEntropy") return BINARYCLASSENTROPY;

		throw std::invalid_argument("[function output_id]: unknown type of output layer");
	}
}