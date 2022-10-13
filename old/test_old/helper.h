//
// Copyright (c) 2022 xsDNN_old Inc. All rights reserved.
//

#ifndef XSDNN_HELPER_H
#define XSDNN_HELPER_H


namespace xsdnn {
    namespace internal {
        namespace debug {
            template <typename Container>
            bool is_near_container(Container c1, Container c2, Scalar tolerance)
            {
                auto i1 = std::begin(c1);
                auto i2 = std::begin(c2);

                for (; i1 != std::end(c1); i1++, i2++)
                {
                    if (std::abs(*i1 - *i2) > tolerance ) return false;
                }
                return true;
            }

            bool is_near_container(Matrix m1, Matrix m2, Scalar tolerance)
            {
                auto i1 = m1.data();
                auto i2 = m2.data();

                for (; i1 != m1.data() + m1.size(); i1++, i2++)
                {
                    if (std::abs(*i1 - *i2) > tolerance ) return false;
                }
                return true;
            }


            bool is_equal_container(Matrix m1, Matrix m2)
            {
                auto i1 = m1.data();
                auto i2 = m2.data();

                for (; i1 != m1.data() + m1.size(); i1++, i2++)
                {
                    if (*i1 != *i2) return false;
                }
                return true;
            }


            bool is_different_container(Matrix m1, Matrix m2)
            {
                auto i1 = m1.data();
                auto i2 = m2.data();

                for (; i1 !=  m1.data() + m1.size(); i1++, i2++)
                {
                    if (*i1 != *i2) return true;
                }
                return false;
            }

            void generate_sinus_data(Matrix& train_image, Matrix& train_label, const Scalar& coef)
            {
                train_image.setRandom();
                train_image *= coef;
                train_label = train_image.array().sin();
            }


            Scalar epsilon      = std::numeric_limits<Scalar>::epsilon();
            Scalar sqrt_epsilon = std::sqrt(std::numeric_limits<Scalar>::epsilon());
        } // namespace debug
    } // namespace internal
} // namespace xsdnn


#endif //XSDNN_HELPER_H
