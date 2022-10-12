//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_MATH_H
#define XSDNN_MATH_H

namespace xsdnn {
    namespace internal {
        namespace math {
            inline void calc_varience(const xsTypes::Matrix &in,
                                      const xsTypes::Vector &mu,
                                      xsTypes::Vector &var) {
                int index = 0;
                for (auto row: in.rowwise()) {
                    Scalar &curr_var = var[index];
                    const auto it = row.begin();
                    const auto en = row.end();
                    const Scalar mx = mu[index++];
                    curr_var = std::accumulate(it, en, curr_var, [mx](Scalar current, Scalar x) {
                        return current + std::pow(x - mx, Scalar(2.0));
                    });
                    curr_var /= row.size();
                }
            }

            inline void update_statistics(const xsTypes::Matrix &in,
                                          const int &dim,
                                          xsTypes::Vector &mu,
                                          xsTypes::Vector &var) {
                mu.resize(dim);
                assert(mu.size() == in.rows());
                mu = in.rowwise().mean();

                var.resize(dim);
                assert(var.size() == in.rows());
                var.setZero();
                calc_varience(in, mu, var);
            }

            /// Получить поколоночный argmax вектор. Каждый элемент arg - argmax в каждом столбце m
            /// \param m матрица для поиска максимумов
            /// \param arg вектор максимумов
            inline void colargmax_vector(const xsTypes::Matrix& m,
                                         std::vector<int>& arg) {
                for (auto col : m.colwise()) {
                    auto max = col.maxCoeff();
                    int index = 0;
                    for (auto num : col) {
                        if (num == max) {
                            arg.push_back(index);
                            break;
                        }
                        index++;
                    }
                }
            }

            /// Получить поколоночный argmax вектор.
            /// \param m матрица для поиска максимумов
            /// \return вектор максимумов
            inline std::vector<int> colargmax_vector(const xsTypes::Matrix& m) {
                std::vector<int> arg;
                for (auto col : m.colwise()) {
                    auto max = col.maxCoeff();
                    int index = 0;
                    for (auto num : col) {
                        if (num == max) {
                            arg.push_back(index);
                            break;
                        }
                        index++;
                    }
                }
                return arg;
            }
        } // namespace math
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_MATH_H
