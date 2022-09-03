//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

/*!
\brief Родительский класс функционала ошибки
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
class Output
{
public:
    virtual ~Output() = default;

    /// <summary>
    /// Здесь проверяем целевую переменную на соотвествие задачи.
    /// 
    /// Например, задача классификации - {0, 1}
    /// </summary>
    /// <param name="target"> - собственно таргет</param>
    virtual void check_target_data(const Matrix& target) const {}

    /// <summary>
    /// Вычисление прямого и обратного распространения для 2 слоев - 
    /// перед выходным и таргетом
    /// </summary>
    /// <param name="prev_layer_data"> - слой перед таргетом</param>
    /// <param name="target"> - собственно таргет</param>
    virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;


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