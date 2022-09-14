//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_MATH_H
#define XSDNN_MATH_H

namespace internal {
    namespace math {
        typedef Eigen::VectorXd Vector;

        inline void calc_varience(const Matrix &in,
                                  const Vector &mu,
                                  Vector &var) {
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

        inline void update_statistics(const Matrix &in,
                                      const int &dim,
                                      Vector &mu,
                                      Vector &var) {
            mu.resize(dim);
            assert(mu.size() == in.rows());
            mu = in.rowwise().mean();

            var.resize(dim);
            assert(var.size() == in.rows());
            var.setZero();
            calc_varience(in, mu, var);
        }
    } // end namespace math
} // end namespace internal

#endif //XSDNN_MATH_H
