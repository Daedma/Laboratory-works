#include <algorithm>
#include <cstring>
#include <array>

#include <fftw3.h>

#include "LightField.hpp"

LightField LightField::fft(size_t newResX, size_t newResY) const
{
	size_t bufferSize = std::max(resX * newResY, newResX * resY);

	// Выделение памяти для FFT
	fftw_complex* buffer = fftw_alloc_complex(bufferSize);

	// Rows
	// Filling
	size_t nNulls = newResX - resX;
	for (size_t i = 0; i != newResY; ++i)
	{
		fftw_complex* dst = buffer + i * newResX;
		const fftw_complex* src = reinterpret_cast<const fftw_complex*>(field.data() + i * resX);
		// Zero padding + shift
		std::memcpy(dst, src + resX / 2, resX / 2 * sizeof(fftw_complex));
		std::memset(dst + resX / 2, 0, nNulls * sizeof(fftw_complex));
		std::memcpy(dst + resX / 2 + nNulls, src, resX / 2 * sizeof(fftw_complex));
	}

	// FFT
	fftw_plan p = fftw_plan_dft_1d(newResX, buffer, buffer, FFTW_FORWARD, FFTW_MEASURE);
	for (size_t i = 0; i != newResY; ++i)
	{
		fftw_execute_dft(p, buffer + i * newResX, buffer + i * newResX);
	}

	// Fit
	for (size_t i = 0; i != newResY; ++i)
	{
		fftw_complex* src = buffer + i * newResX;
		fftw_complex* dst = buffer + i * resX;
		std::memcpy(dst + resX / 2, src, resX / 2 * sizeof(fftw_complex));
		std::memcpy(dst, src + newResX - resX / 2, resX / 2 * sizeof(fftw_complex));
	}

	// Transpose
	for (size_t i = 0; i != resY; ++i)
	{
		for (size_t j = i; j != resX; ++j)
		{
			std::swap(buffer[i + j * resX], buffer[j + i * resX]);
		}
	}



}
