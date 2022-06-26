//
// Created by shuffle on 25.06.22.
//

// https://habr.com/ru/post/263993/

#ifndef XSDNN_INCLUDE_NORMAL_H
#define XSDNN_INCLUDE_NORMAL_H

# include "../RNG.h"
# include "Exponential.h"

static Scalar stairWidthNormal[257], stairHeightNormal[256];

class Normal
{
private:
    constexpr static const Scalar x1 = 3.6541528853610088;
    constexpr static const Scalar A = 4.92867323399e-3;

    static Scalar Uniform(Scalar a, Scalar& b, RNG& rng) {
        return a + rng.rand() * (b - a) / RAND_MAX;
    }

    static void setupNormalTables() {
        stairHeightNormal[0] = std::exp(-.5 * x1 * x1);
        stairWidthNormal[0] = A / stairHeightNormal[0];
        stairWidthNormal[256] = 0;
        for (unsigned i = 1; i <= 255; ++i)
        {
            stairWidthNormal[i] = std::sqrt(-2 * std::log(stairHeightNormal[i - 1]));
            stairHeightNormal[i] = stairHeightNormal[i - 1] + A / stairWidthNormal[i];
        }
    }

    static Scalar NormalZiggurat(RNG& rng) {
        int iter = 0;
        do {
            Scalar B = rng.rand();
            int stairId = static_cast<int>(B) & 255;
            Scalar x = Uniform(0, stairWidthNormal[stairId], rng);
            if (x < stairWidthNormal[stairId + 1])
                return ((signed)B > 0) ? x : -x;
            if (stairId == 0)
            {
                static Scalar z = -1;
                Scalar y;
                if (z > 0)
                {
                    x = internal::get_exponential(x1, rng);
                    z -= 0.5 * x * x;
                }
                if (z <= 0)
                {
                    do {
                        x = internal::get_exponential(x1, rng);
                        y = internal::get_exponential(1, rng);
                        z = y - 0.5 * x * x;
                    } while (z <= 0);
                }
                x += x1;
                return ((signed)B > 0) ? x : -x;
            }

            if (Uniform(stairHeightNormal[stairId - 1], stairHeightNormal[stairId], rng) < std::exp(-.5 * x * x))
                return ((signed)B > 0) ? x : -x;
        } while (++iter <= 1e9);
        return NAN;
    }

    static void check_distribution_param(const std::vector<Scalar>& params)
    {
        if (params.size() != 2) throw std::length_error("[class Normal] Normal distribution have 2 params."
                                                        " Check input data.");

        if (params[1] < 0) throw std::invalid_argument("[class Normal] Variance must be equal than zero."
                                                       " Check input data");
    }
public:
    /// Заполнить массив случайными числами из нормального распределения с параметрами \mu и \sigma
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

        setupNormalTables();

        const Scalar mu = params[0];
        const Scalar sigma = params[1];

        for (int i = 0; i < n; i++)
        {
            arr[i] = mu + NormalZiggurat(rng) * sigma;
        }
    }

    ///
    /// \return тип распределения
    static std::string return_type() { return "Normal"; }
};
#endif //XSDNN_INCLUDE_NORMAL_H
