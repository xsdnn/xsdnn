//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

// https://habr.com/ru/post/263993/

#ifndef XSDNN_INCLUDE_EXPONENTIAL_H
#define XSDNN_INCLUDE_EXPONENTIAL_H

#include "../Utils/Except.h"

namespace xsdnn {
    namespace init {
        static Scalar stairWidth[257], stairHeight[256];

        /*!
        \brief Класс генерации экспоненциального (Exponential) распределения
        \author __[shuffle-true](https://github.com/shuffle-true)__
        \version 0.0
        \date Июнь 2022 года
        */
        class Exponential {
        private:
            constexpr static const Scalar x1 = 7.69711747013104972;
            constexpr static const Scalar A = 3.9496598225815571993e-3;

            static Scalar Uniform(Scalar a, Scalar &b, RNG &rng) {
                return a + rng.rand() * (b - a) / RAND_MAX;
            }

            static void setupExpTables() {
                stairHeight[0] = std::exp(-x1);
                stairWidth[0] = A / stairHeight[0];
                stairWidth[256] = 0;
                for (unsigned i = 1; i <= 255; ++i) {
                    stairWidth[i] = -std::log(stairHeight[i - 1]);
                    stairHeight[i] = stairHeight[i - 1] + A / stairWidth[i];
                }
            }

            static Scalar ExpZiggurat(RNG &rng) {
                int iter = 0;
                do {
                    int stairId = static_cast<int>(rng.rand()) & 255;
                    Scalar x = Uniform(0, stairWidth[stairId], rng);
                    if (x < stairWidth[stairId + 1])
                        return x;
                    if (stairId == 0)
                        return x1 + ExpZiggurat(rng);
                    if (Uniform(stairHeight[stairId - 1], stairHeight[stairId], rng) < std::exp(-x))
                        return x;
                } while (++iter <= 1e9);
                return NAN;
            }

            static void check_distribution_param(const std::vector<Scalar> &params) {
                if (params.size() != 1)
                    throw internal::except::xs_error("[class Exponential] Exponential distribution have 1 params."
                                                     " Check input data.");

                if (params[0] == 0)
                    throw internal::except::xs_error("[class Exponential] Exponential distribution have 1 equal then "
                                                     "zero params. Check input data.");

            }

        public:
            /// Заполнить массив случайными числами из экспоненциального распределения с параметром \lambda
            /// \param arr указатель на массив
            /// \param n размер массива
            /// \param rng ГСЧ
            /// \param params вектор параметров распределения
            static void set_random_data(Scalar *arr,
                                        const int n,
                                        RNG &rng,
                                        const std::vector<Scalar> &params) {
                check_distribution_param(params);

                setupExpTables();

                const Scalar rate = params[0];

                for (int i = 0; i < n; i++) {
                    arr[i] = ExpZiggurat(rng) / rate;
                }
            }

            Scalar exponential(const Scalar &rate, RNG &rng) {
                return ExpZiggurat(rng) / rate;
            }

            ///
            /// \return тип распределения
            static std::string return_type() { return "Exponential"; }
        };
    }


    namespace internal {
        /// Получить одно значения из экспоненциального распределения
        /// \param rate параметр распределения
        /// \param rng ГСЧ
        /// \return
        Scalar get_exponential(const Scalar &rate, RNG &rng) {
            init::Exponential exp;
            return exp.exponential(rate, rng);
        }
    } // namespace init
} // namespace xsdnn


#endif //XSDNN_INCLUDE_EXPONENTIAL_H
