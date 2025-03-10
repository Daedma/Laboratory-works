#include <algorithm>
#include <cstring>
#include <array>
#include <algorithm>
#include <iostream>

#include <fftw3.h>

#include "LightField.hpp"

template<typename T>
static int m1pn(T n)
{
	return 1 - 2 * (n & 1);
}

static size_t nextPowerOf2(size_t n)
{
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n |= n >> 32;
	++n;
	return n;
}

LightField LightField::fft_Impl(int dftDirection) const
{	

	size_t N = res;
	size_t M = nextPowerOf2(res) << 1;

	size_t bufferSize = M * N;

	// Выделение памяти для FFT
	fftw_complex* buffer = fftw_alloc_complex(bufferSize);

	// Rows
	shift(get(), N, N, buffer, M, M);

	fftw_plan p = fftw_plan_dft_1d(M, buffer, buffer, dftDirection, FFTW_MEASURE);
	for (size_t i = 0; i != N; ++i)
	{
		fftw_execute_dft(p, buffer + i * M, buffer + i * M);
	}

	fit(buffer, N, M, buffer + (M - N) / 2, M, M);

	// C0oumns
	transpose(buffer + (M - N) / 2, N, M);

	shift(buffer + (M - N) / 2, N, M, buffer, M, M);

	for (size_t i = 0; i != N; ++i)
	{
		fftw_execute_dft(p, buffer + i * M, buffer + i * M);
	}

	fit(buffer, N, M, buffer + (M - N) / 2, M, M);

	transpose(buffer + (M - N) / 2, N, M);

	const double newSize = N * N / (2. * size * M);

	LightField result{ newSize, N};
	for (size_t i = 0; i != N; ++i)
	{
		fftw_complex* dst = result.get(0, i);
		const fftw_complex* src = buffer + i * M;
		std::memcpy(dst, src, N * sizeof(fftw_complex));
	}

	double hh = res * res / (size * size);
	for(auto& i : result.field)
	{
		i *= hh;
	}

	fftw_free(buffer);
	fftw_destroy_plan(p);

	return result;
}


void LightField::shift(const fftw_complex* src, size_t srcSize, 
	size_t srcStep,  fftw_complex* dst, size_t dstSize, size_t dstStep) const
{
	size_t nNulls = dstSize - srcSize;
	for (size_t i = 0; i != srcSize; ++i)
	{
		// Zero padding + shift
		std::memcpy(dst, src + srcSize / 2, srcSize / 2 * sizeof(fftw_complex));
		std::memcpy(dst + srcSize / 2 + nNulls, src, srcSize / 2 * sizeof(fftw_complex));
		std::memset(dst + srcSize / 2, 0, nNulls * sizeof(fftw_complex));
		dst += dstStep;
		src += srcStep;
	}
}

void LightField::fit(const fftw_complex* src, size_t srcSize, 
	size_t srcStep,  fftw_complex* dst, size_t dstSize, size_t dstStep) const
{
	for (size_t i = 0; i != srcSize; ++i)
	{
		std::memcpy(dst + srcSize / 2, src, srcSize / 2 * sizeof(fftw_complex));
		std::memcpy(dst, src + dstSize - srcSize / 2, srcSize / 2 * sizeof(fftw_complex));
		dst += dstStep;
		src += srcStep;
	}
}

void LightField::transpose(fftw_complex* src, size_t srcSize, size_t srcStep) const
{
	for (size_t i = 0; i != srcSize; ++i)
	{
		for (size_t j = i; j != srcSize; ++j)
		{
			fftw_complex* rhs = src + i * srcStep + j;
			fftw_complex* lhs = src + j * srcStep + i;
			std::swap((*rhs)[0], (*lhs)[0]);
			std::swap((*rhs)[1], (*lhs)[1]);
		}
	}
}