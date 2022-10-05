//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_SGD_H
#define XSDNN_SGD_H

namespace xsdnn {
    /*!
\brief Класс стохастического градиентного спуска (SGD)
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Июль 2022 года
*/
    namespace optim {

        class SGD : public Optimizer {
        private:
            typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Vector;
            std::map<const Scalar *, Vector> momentum_history;

        public:
            Scalar m_lrate;     ///< длина шага.
            Scalar m_decay;     ///< коэффицент линейной зависимости с производной.
            Scalar m_momentum;  ///< импульс спуска
            Scalar m_dampening; ///< сглаживание момента
            bool m_nesterov;  ///< активация момента Нестерова

            ///
            /// \param lrate длина шага.
            /// \param decay weight_decay L2-penalty.
            /// \param momentum momentum factor (default: 0)
            /// \param dampening dampening for momentum (default: 0)
            /// \param nesterov enables Nesterov momentum (default: false)
            explicit SGD(const Scalar &lrate = Scalar(0.01), const Scalar &decay = Scalar(0),
                         const Scalar &momentum = Scalar(0.0), const bool &nesterov = false,
                         const Scalar &dampening = Scalar(0.0)) :
                    m_lrate(lrate), m_decay(decay), m_momentum(momentum), m_dampening(dampening), m_nesterov(nesterov) {}

            /// Сброс истории момента
            void reset() override {
                momentum_history.clear();
            }

            /// \image html sgd_implementation.jpg
            /// \param grad вектор производной (например веса или смещения)
            /// \param theta вектор значений (например веса или смещения)
            void update(AlignedMapVec &grad, AlignedMapVec &theta) override {
                if (m_decay != 0) grad += m_decay * theta;

                if (m_momentum != 0) {
                    Vector &beta = momentum_history[grad.data()];

                    if (beta.size() != 0) {
                        beta = m_momentum * beta + (1 - m_dampening) * grad.array();
                    } else {
                        beta = grad;
                    }

                    if (m_nesterov) {
                        grad.array() = grad.array() + m_momentum * beta;
                    } else {
                        grad.array() = beta;
                    }
                }
                theta.noalias() -= m_lrate * grad;
            }
        };

        class SGDAdaptive : public SGD {
        private:
            const size_t& iter_per_epoch_;
            const size_t& step_size_;
            const Scalar& lr_gamma_;
            const Scalar& decay_gamma_;
            const Scalar& momentum_gamma_;
            const Scalar& dampening_gamma_;
            size_t curr_iter_;

        public:
            explicit SGDAdaptive(
                    const size_t& iter_per_epoch,
                    const size_t& step_size = 1,
                    const Scalar &lrate = Scalar(0.01),
                    const Scalar& lr_gamma = Scalar(1),
                    const Scalar &decay = Scalar(0),
                    const Scalar& decay_gamma = Scalar(1),
                    const Scalar &momentum = Scalar(0.0),
                    const Scalar& momentum_gamma = Scalar(1),
                    const Scalar &dampening = Scalar(0.0),
                    const Scalar& dampening_gamma = Scalar(1),
                    const bool &nesterov = false
            ) :
                    SGD(lrate, decay, momentum, nesterov, dampening),
                    iter_per_epoch_(iter_per_epoch),
                    step_size_(step_size),
                    lr_gamma_(lr_gamma),
                    decay_gamma_(decay_gamma),
                    momentum_gamma_(momentum_gamma),
                    dampening_gamma_(dampening_gamma),
                    curr_iter_(0)
            {}

            void update(AlignedMapVec &grad, AlignedMapVec &theta) {
                SGD::update(grad, theta);
                curr_iter_ += 1;

                if (curr_iter_ / iter_per_epoch_ == step_size_) {
                    curr_iter_ = 0;
                    m_lrate *= lr_gamma_;
                    m_decay *= decay_gamma_;
                    m_momentum *= momentum_gamma_;
                    m_dampening *= dampening_gamma_;
                }
            }
        };
    } // namespace optim
} // namespace xsdnn


#endif //XSDNN_SGD_H
