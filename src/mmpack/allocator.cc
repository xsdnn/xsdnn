//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#include "allocator.h"
#include "../config.h"

namespace mmpack {

    namespace detail {
        MM_STRONG_INLINE void* _aligned_malloc(std::size_t size, std::size_t alignment)
        {
            void* res = 0;
            void* ptr = malloc(size+alignment);
            if(ptr != 0)
            {
                res = reinterpret_cast<void*>((reinterpret_cast<size_t>(ptr) & ~(size_t(alignment-1))) + alignment);
                *(reinterpret_cast<void**>(res) - 1) = ptr;
            }
            return res;
        }

        MM_STRONG_INLINE void _aligned_free(void* ptr)
        {
            if(ptr != 0)
                free(*(reinterpret_cast<void**>(ptr)-1));
        }
    } // detail

    template<typename T, std::size_t alignment>
    T* aligned_allocator<T, alignment>::address(T& x) const noexcept {
        return &x;
    }


    template<typename T, std::size_t alignment>
    const T* aligned_allocator<T, alignment>::address(const T& x) const noexcept {
        return &x;
    }

    template<typename T, std::size_t alignment>
    T* aligned_allocator<T, alignment>::allocate(std::size_t n, const void *hint) {
        T* res = reinterpret_cast<T*>(aligned_malloc(sizeof(T)*n));
        if (res == 0) {
            throw std::bad_alloc();
        }
        return res;
    }

    template<typename T, std::size_t alignment>
    void aligned_allocator<T, alignment>::deallocate(T* p, std::size_t n) {
        aligned_free(p);
    }

    template<typename T, std::size_t alignment>
    std::size_t aligned_allocator<T, alignment>::max_size() const noexcept {
        return std::size_t(-1) / sizeof(T);
    }

    template<typename T, std::size_t alignment>
    MM_STRONG_INLINE void* aligned_allocator<T, alignment>::aligned_malloc(size_type size) {
#if HAS_MM_MALLOC
        return _mm_malloc(size, alignment);
#elif HAS_POSIX_MEMALIGN
        void* r;
    const int32_t fail = posix_memalign(&r, size, alignment);
    if (fail) res = 0;
    return res;
#else
    return detail::_aligned_malloc(size, alignment);
#endif
    }

    template<typename T, std::size_t alignment>
    MM_STRONG_INLINE void aligned_allocator<T, alignment>::aligned_free(void *p) {
#if HAS_MM_MALLOC
        _mm_free(p);
#elif HAS_POSIX_MEMALIGN
        free(p);
#else
    detail::_aligned_free(p);
#endif
    }

} // mmpack

template class mmpack::aligned_allocator<mmpack::mm_scalar, 64>;
