#pragma once

# include <Eigen/Core>
# include <stdexcept>
# include "Config.h"


///
/// Интерфейс выходного слоя сетки.
/// 


class Output
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::RowVectorXi IntegerVector;

public:
    virtual ~Output() {}

    /// <summary>
    /// Здесь проверяем целевую переменную на соотвествие задачи.
    /// 
    /// Например, задача классификации - {0, 1}
    /// </summary>
    /// <param name="target"> - собственно таргет</param>
    virtual void check_target_data(const Matrix& target) {}

    /// <summary>
    /// Отдельный блок проверки для многоклассовой классификации.
    /// Может не подойти для регрессии поэтому сразу вызовем исключение 
    /// </summary>
    /// <param name="target"> - собственно таргет</param>
    virtual void check_target_data(IntegerVector& target)
    {
        throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
    }

    /// <summary>
    /// Вычисление прямого и обратного распространения для 2 слоев - 
    /// перед выходным и таргетом
    /// </summary>
    /// <param name="prev_layer_data"> - слой перед таргетом</param>
    /// <param name="target"> - собственно таргет</param>
    virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

    /// <summary>
    /// Вычисление прямого и обратного распространения для 2 слоев 
    /// (Для многоклассовой классификации, !изначально выводится исключение!)
    ///  - 
    /// перед выходным и таргетом 
    /// </summary>
    /// <param name="prev_layer_data"></param>
    /// <param name="target"></param>
    virtual void evaluate(const Matrix& prev_layer_data,
        const IntegerVector& target)
    {
        throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
    }

    virtual const Matrix& backprop_data() const = 0;

    /// <summary>
    /// None
    /// </summary>
    /// <returns>Вычисляет лосс</returns>
    virtual Scalar loss() const = 0;

    /// <summary>
    /// None
    /// </summary>
    /// <returns>Тип выходного слоя, нужно для записи модели в файл</returns>
    virtual std::string output_type() const = 0;
};