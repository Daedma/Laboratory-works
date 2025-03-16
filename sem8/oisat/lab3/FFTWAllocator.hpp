#pragma once

#include <fftw3.h>

template <typename T>
class FFTWAllocator
{
public:
	using value_type = T;

	FFTWAllocator() = default;

	template <typename U>
	FFTWAllocator(const FFTWAllocator<U>&) {}

	T* allocate(std::size_t n)
	{
		return static_cast<T*>(fftw_malloc(sizeof(T) * n));
	}

	void deallocate(T* p, std::size_t)
	{
		fftw_free(p);
	}
};

template <typename T, typename U>
bool operator==(const FFTWAllocator<T>&, const FFTWAllocator<U>&)
{
	return true;
}

template <typename T, typename U>
bool operator!=(const FFTWAllocator<T>&, const FFTWAllocator<U>&)
{
	return false;
}