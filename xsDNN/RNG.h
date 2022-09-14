//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_RNG_H
#define XSDNN_RNG_H

# include "Config.h"

/*!
\brief Класс ГСЧ
\author __[shuffle-true](https://github.com/shuffle-true)__
\version 0.0
\date Март 2022 года
*/
class RNG {
private:
    const unsigned int m_a; // multiplier
    const unsigned long m_max; // 2^31 - 1
    long m_rand;

    inline long next_long_rand(long seed) const {
        unsigned long lo, hi;
        lo = m_a * (long) (seed & 0xFFFF);
        hi = m_a * (long) ((unsigned long) seed >> 16);
        lo += (hi & 0x7FFF) << 16;

        if (lo > m_max) {
            lo &= m_max;
            ++lo;
        }

        lo += hi >> 15;

        if (lo > m_max) {
            lo &= m_max;
            ++lo;
        }

        return (long) lo;
    }

public:
    explicit RNG(unsigned long init_seed, const bool divide_m_max = true) :
            m_a(16807),
            m_max(2147483647L),
            m_rand(init_seed ? (init_seed & m_max) : 1) {}

    ~RNG() = default;

    void seed(unsigned long seed) {
        m_rand = (seed ? (seed & m_max) : 1);
    }

    Scalar rand() {
        m_rand = next_long_rand(m_rand);
        return Scalar(m_rand);
    }
};

#endif //XSDNN_RNG_H
