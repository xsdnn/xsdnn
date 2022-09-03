//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

# include <random>

namespace internal {
    namespace random {
        /// Рандомизация массива
        /// \param arr массив
        /// \param n размер массива
        /// \param rng генератор
        inline void shuffle(int* arr, const int n, RNG& rng)
        {
            for (int i = n - 1; i > 0; i--)
            {
                const int j = int(rng.rand() * (i + 1) / RAND_MAX);

                const int tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }

        /// Генерация перетасованных батчей
        /// \tparam DerivedX
        /// \tparam DerivedY
        /// \tparam XType тип данных на выборке
        /// \tparam YType тип данных на целевой переменной
        /// \param x обучающая выборка
        /// \param y целевая переменная
        /// \param batch_size размер батча
        /// \param rng генератор
        /// \param x_batches полученные батчи для выборки
        /// \param y_batches полученные батчи для целевой переменной
        /// \return
        template <typename DerivedX, typename DerivedY, typename XType, typename YType>
        inline int create_shuffled_batches(
                const Eigen::MatrixBase<DerivedX>& x, const Eigen::MatrixBase<DerivedY>& y,
                int batch_size, RNG& rng,
                std::vector<XType>& x_batches, std::vector<YType>& y_batches
        )
        {
            const int nobs = x.cols();
            const int dimx = x.rows();
            const int dimy = y.rows();

            if (y.cols() != nobs)
            {
                throw std::invalid_argument("Input X and Y have different number of observations");
            }

            Eigen::VectorXi id = Eigen::VectorXi::LinSpaced(nobs, 0, nobs - 1);
            shuffle(id.data(), id.size(), rng);

            if (batch_size > nobs)
            {
                batch_size = nobs;
            }

            const int nbatch = (nobs - 1) / batch_size + 1;
            const int last_batch_size = nobs - (nbatch - 1) * batch_size;
            x_batches.clear();
            y_batches.clear();
            x_batches.reserve(nbatch);
            y_batches.reserve(nbatch);

            for (int i = 0; i < nbatch; i++)
            {
                const int bsize = (i == nbatch - 1) ? last_batch_size : batch_size;
                x_batches.push_back(XType(dimx, bsize));
                y_batches.push_back(YType(dimy, bsize));
                const int offset = i * batch_size;

                for (int j = 0; j < bsize; j++)
                {
                    x_batches[i].col(j).noalias() = x.col(id[offset + j]);
                    y_batches[i].col(j).noalias() = y.col(id[offset + j]);
                }
            }

            return nbatch;
        }

        /// Генерация нормального распределения через алгоритм __Бокса-Мюллера__
        /// \param arr массив
        /// \param n размер массива
        /// \param rng генератор
        /// \param mu мат.ожидание
        /// \param sigma дисперсия
        inline void set_normal_random(
                Scalar* arr,
                const int n,
                RNG& rng,
                const Scalar& mu = Scalar(0),
                const Scalar& sigma = Scalar(1))
        {

            const double two_pi = 6.283185307179586476925286766559;

            for (int i = 0; i < n - 1; i += 2)
            {
                const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
                const double t2 = two_pi * rng.rand();
                arr[i] = t1 * std::cos(t2) + mu;
                arr[i + 1] = t1 * std::sin(t2) + mu;
            }

            if (n % 2 == 1)
            {
                const double t1 = sigma * std::sqrt(-2 * std::log(rng.rand()));
                const double t2 = two_pi * rng.rand();
                arr[n - 1] = t1 * std::cos(t2) + mu;
            }
        }

        /// Генерация массива из равномерного распределения
        /// \param arr массив
        /// \param n размер массива
        /// \param rng генератор натуральный чисел в диапазоне от 0 до RAND_MAX
        /// \param a левая граница распределения
        /// \param b правая граница распределения
        inline void set_uniform_random(
                Scalar* arr,
                const int n,
                RNG& rng,
                const Scalar& a = Scalar(0),
                const Scalar& b = Scalar(1))
        {
            const Scalar coefficent_ = (b - a);

            for (int i = 0; i < n; i++)
            {
                arr[i] = a + rng.rand() * coefficent_;
            }
        }

        inline Scalar set_uniform_random(
                RNG& rng,
                const Scalar& a = Scalar(0),
                const Scalar& b = Scalar(1))
        {
            return a + rng.rand() * (b - a) / RAND_MAX;
        }

        /*!
        \brief Класс генерации распределения Бернулли
        \author __[shuffle-true](https://github.com/shuffle-true)__
        \version 0.0
        \date Июль 2022 года
        */
        class bernoulli
        {
        private:
            RNG                 rng_;               ///< ГСЧ
            Scalar              p_value_;           ///< вероятность __отключения__ нейрона
            long                array_size_;        ///< размер колонки маски (экв. кол-во признаков у объекта)
            std::random_device  rd;                 ///< движок для недетерменированной генерации зерна

        public:
            explicit bernoulli() : rng_(42), p_value_(-1.0), array_size_(-1) {}

            /// Установить параметры генерации распределения
            /// \param p_value вероятность __отключения__ нейрона
            /// \param array_size размер колонки маски (экв. кол-во признаков у объекта)
            void set_param(const Scalar& p_value, const long& array_size)
            {
                if (p_value_ != -1.0 && array_size_ != -1) return;

                this->p_value_ = p_value;
                this->array_size_ = array_size;
            }

            ///
            /// \param arr указатель на массив маски
            inline void operator () (Scalar* arr)
            {
                // non-determenistic random generator for dropout mask
#if defined(DNN_NO_DTRMINIST)
                rng_.seed(rd());
#else
                rng_.seed(42);
#endif

                for (long i = 0; i < array_size_; i++)
                {
                    arr[i] = static_cast<Scalar>(set_uniform_random(rng_) <= 1 - p_value_);
                }
            }

            /// Использование метода не предусмотрено.
            /// \return 1 || 0
            inline int operator () ()
            {
                return static_cast<int>(set_uniform_random(rng_) <= p_value_);
            }

            ///
            /// \return p_value_
            Scalar p() const
            {
                return p_value_;
            }
        };
    } // end namespace random
} // end namespace internal

namespace internal
{

}
