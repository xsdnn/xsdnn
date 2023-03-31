//
// Created by Andrei R. on 31.12.22.
// Copyright (c) 2022 xsDNN. All rights reserved.
//

#ifndef XSDNN_RANDOM_HPP
#define XSDNN_RANDOM_HPP

namespace xsdnn {

class base_distribution {
public:
    virtual ~base_distribution() {}

protected:
    std::random_device rd;
};

template<typename T>
class uniform_distribution : protected base_distribution {
public:
    uniform_distribution(T min, T max) : min_(min), max_(max), coef_(max_ - min_) {}

#if defined(XS_NO_DTRMNST)
    T operator() (base_random_engine& eng) {
        return min_ + eng.rand(rd) * coef_ / RAND_MAX;
    }
#else
    T operator() (base_random_engine& eng) {
        return min_ + eng.rand() * coef_ / RAND_MAX;
    }
#endif

private:
    T min_;
    T max_;
    T coef_;
};

template <typename T>
T uniform_rand(T min, T max) {
    uniform_distribution<T> dst(min, max);
    default_random_engine eng;
    return dst(eng);
}

template<typename T>
void uniform_rand(T* data, size_t size, T min, T max) {
    uniform_distribution<T> dst(min, max);
    default_random_engine eng;
    for (size_t i = 0; i < size; ++i) {
        data[i] = dst(eng);
    }
}

} // xsdnn

#endif //XSDNN_RANDOM_HPP
