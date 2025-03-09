#include <algorithm>
#include <cstring>
#include <array>
#include <algorithm>

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

LightField LightField::fft_Impl(size_t newResX, size_t newResY, int dftDirection) const
{
	if (newResX == 0 || newResY == 0)
	{
		throw std::invalid_argument("Resolution must be greater than zero.");
	}

	newResX = nextPowerOf2(newResX);
	newResY = nextPowerOf2(newResY);

	const size_t bufferSize = newResX * newResY;
	fftw_complex* buffer = fftw_alloc_complex(bufferSize);
	std::memset(buffer, 0, bufferSize * sizeof(fftw_complex));

	for (size_t i = 0; i != resY; ++i)
	{
		for (size_t j = 0; j != resX; ++j)
		{
			const fftw_complex* cur = get(j, i);
			int sign = m1pn(i + j);
			buffer[i * newResX + j][0] = (*cur)[0] * sign;
			buffer[i * newResX + j][1] = (*cur)[1] * sign;
		}
	}

	fftw_plan plan = fftw_plan_dft_2d(newResY, newResX, buffer, buffer, dftDirection, FFTW_ESTIMATE);
	fftw_execute(plan);

	const double newSizeX = 2 * resX * resX / (4 * sizeX * newResX);
	const double newSizeY = 2 * resY * resY / (4 * sizeY * newResY);

	LightField result{ newSizeX, newSizeY, resX, resY };
	for (size_t i = 0; i != resY; ++i)
	{
		fftw_complex* dst = result.get(0, i);
		const fftw_complex* src = buffer + i * newResX;
		std::memcpy(dst, src, resX * sizeof(fftw_complex));
	}

	double hx = sizeX / resX;
	double hy = sizeY / resY;
	for(auto& i : result.field)
	{
		i *= hx * hy;
	}

	fftw_free(buffer);
	fftw_destroy_plan(plan);

	return result;
}
