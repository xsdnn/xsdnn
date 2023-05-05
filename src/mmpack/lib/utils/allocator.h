//
// Created by rozhin on 01.05.23.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef MMPACK_ALLOCATOR_H
#define MMPACK_ALLOCATOR_H

#include <cstddef>
#include <exception>
#include "macro.h"


namespace mmpack {


    template<typename T, std::size_t alignment>
    class aligned_allocator {
    public:
        typedef T value_type;
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef std::size_t size_type;
        typedef ptrdiff_t difference_type;

        template <typename U>
        struct rebind {
            typedef aligned_allocator<U, alignment> other;
        };

        aligned_allocator() = default;

        pointer address(reference x) const noexcept;
        const_pointer address(const_reference x) const noexcept;

        pointer allocate(size_type n, const void* hint = 0);
        void deallocate(pointer p, size_type);

        size_type max_size() const noexcept;

        template<class U, class... Args>
        void construct(U* ptr, Args&&... args) {
            void *p = ptr;
            ::new (p) U(std::forward<Args>(args)...);
        }

        template <class U>
        void construct(U *ptr) {
            void *p = ptr;
            ::new (p) U();
        }

        template<class U>
        void destroy(U* ptr) {
            ptr->~U();
        }

    private:
        MM_STRONG_INLINE void* aligned_malloc(size_type size);
        MM_STRONG_INLINE void  aligned_free(void* p);
    };

    template <typename T1, typename T2, std::size_t alignment>
    inline bool operator==(const aligned_allocator<T1, alignment> &,
                           const aligned_allocator<T2, alignment> &) {
        return true;
    }

    template <typename T1, typename T2, std::size_t alignment>
    inline bool operator!=(const aligned_allocator<T1, alignment> &,
                           const aligned_allocator<T2, alignment> &) {
        return false;
    }


} // mmpack

#endif //MMPACK_ALLOCATOR_H
