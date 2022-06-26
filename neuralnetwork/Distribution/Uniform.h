//
// Created by shuffle on 25.06.22.
//

// https://habr.com/ru/post/263993/

#ifndef XSDNN_INCLUDE_UNIFORM_H
#define XSDNN_INCLUDE_UNIFORM_H

# include <vector>
# include <stdexcept>
# include <string>
# include "../RNG.h"

class Uniform
{
private:
    static Scalar get_uniform_distribution(const Scalar& a, const Scalar coef_, RNG& rng)
    {
        return a + rng.rand() * coef_ / RAND_MAX;
    }

    static void check_distribution_param(const std::vector<Scalar>& params)
    {
        if (params.size() != 2) throw std::length_error("[class Uniform] Uniform distribution have 2 params."
                                                        " Check input data.");
        if (params[0] == params[1]) throw std::invalid_argument("[class Uniform] Uniform distribution have 2 difference "
                                                                "params. Check input data");
    }
public:
    /// Заполнить массив случайными числами из равномерного распределения на интервале [a, b]
    /// \param arr указатель на массив
    /// \param n размер массива
    /// \param rng ГСЧ
    /// \param params вектор параметров распределения
    static void set_random_data(Scalar* arr,
                         const int n,
                         RNG& rng,
                         const std::vector<Scalar>& params)
    {
        check_distribution_param(params);

        const Scalar a = params[0];
        const Scalar b = params[1];

        const Scalar coef_ = (b - a);

        for (int i = 0; i < n; i++)
        {
            arr[i] = get_uniform_distribution(a, coef_, rng);
        }
    }

    ///
    /// \return тип распределения
    static std::string return_type()
    {
        return "Uniform";
    }
};

#endif //XSDNN_INCLUDE_UNIFORM_H
