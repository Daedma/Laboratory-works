#pragma once

#include <type_traits>
#include <fftw3.h>
#include <complex>

template <typename T>
struct FFTWAllocator
{
	typedef T value_type;

	FFTWAllocator() = default;

	template<class T>
	constexpr FFTWAllocator(const FFTWAllocator<T>&) noexcept {}

	T* allocate(size_t n) = delete;

	void deallocate(T* p, size_t n) noexcept
	{
		fftw_free(p);
	}
};

template<class T, class U>
bool operator==(const FFTWAllocator<T>&, const FFTWAllocator<U>&) { return true; }

template<class T, class U>
bool operator!=(const FFTWAllocator<T>&, const FFTWAllocator<U>&) { return false; }

template <>
struct FFTWAllocator<std::complex<double>>
{
	std::complex<double>* allocate(size_t n)
	{
		return reinterpret_cast<std::complex<double>*>(fftw_alloc_complex(n));
	}
};