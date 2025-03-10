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
	const size_t N = res;
	const size_t M = nextPowerOf2(nextPowerOf2(N) + 1); // Дополнение нулями до размерности M
	const double h = size / N;

	fftw_complex* buffer = fftw_alloc_complex(M * M);
	std::memset(buffer, 0, M * M * sizeof(fftw_complex));

	const double normalisation = h * h;

	size_t startIndex = (M - N) / 2;
	for (size_t i = 0; i != N; ++i)
	{
		for (size_t j = 0; j != N; ++j)
		{
			const fftw_complex* cur = get(j, i);
			fftw_complex* dst = buffer + (i + startIndex) * M + (j + startIndex);
			(*dst)[0] = (*cur)[0] * normalisation;
			(*dst)[1] = (*cur)[1] * normalisation;
		}
	}

	// std::cout << "PRINT BEGIN\n";
	// for (size_t i = 0; i != M; ++i)
	// {
	// 	for (size_t j = 0; j != M; ++j)
	// 	{
	// 		fftw_complex* cur = buffer + i * M + j;
	// 		std::cout << "(" << (*cur)[0] << ";" << (*cur)[1] << ") ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "PRINT END\n";

	for (size_t i = 0; i != N / 2; ++i)
	{
		fftw_complex* srcUpLeft = buffer + (i + startIndex) * M + startIndex;
		fftw_complex* srcUpRight = srcUpLeft + N / 2;
		fftw_complex* srcDownLeft = srcUpLeft + N / 2 * M;
		fftw_complex* srcDownRight = srcDownLeft + N / 2;

		fftw_complex* dstUpLeft = buffer + i * M;
		fftw_complex* dstUpRight = dstUpLeft + M - N / 2;
		fftw_complex* dstDownLeft = dstUpLeft + (M - N / 2) * M;
		fftw_complex* dstDownRight = dstDownLeft + M - N / 2;

		size_t copySize = N / 2 * sizeof(fftw_complex);
		std::memcpy(dstUpLeft, srcDownRight, copySize);
		std::memcpy(dstUpRight, srcDownLeft, copySize);
		std::memcpy(dstDownLeft, srcUpRight, copySize);
		std::memcpy(dstDownRight, srcUpLeft, copySize);
	}

	// std::cout << "PRINT BEGIN\n";
	// for (size_t i = 0; i != M; ++i)
	// {
	// 	for (size_t j = 0; j != M; ++j)
	// 	{
	// 		fftw_complex* cur = buffer + i * M + j;
	// 		std::cout << "(" << (*cur)[0] << ";" << (*cur)[1] << ") ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "PRINT END\n";

	fftw_plan plan = fftw_plan_dft_2d(M, M, buffer, buffer, dftDirection, FFTW_ESTIMATE);
	fftw_execute(plan);

	const double newSize = 2 * (N * N / (4 * (size * 0.5) * M));

	LightField result{ newSize, N };

	for (size_t i = 0; i != N / 2; ++i)
	{
		fftw_complex* srcUpLeft = buffer + i * M;
		fftw_complex* srcUpRight = srcUpLeft + M - N / 2;
		fftw_complex* srcDownLeft = srcUpLeft + (M - N / 2) * M;
		fftw_complex* srcDownRight = srcDownLeft + M - N / 2;

		fftw_complex* dstUpLeft = result.get(0, i);
		fftw_complex* dstUpRight = result.get(N / 2, i);
		fftw_complex* dstDownLeft = result.get(0, i + N / 2);
		fftw_complex* dstDownRight = result.get(N / 2, i + N / 2);

		size_t copySize = N / 2 * sizeof(fftw_complex);
		std::memcpy(dstUpLeft, srcDownRight, copySize);
		std::memcpy(dstUpRight, srcDownLeft, copySize);
		std::memcpy(dstDownLeft, srcUpRight, copySize);
		std::memcpy(dstDownRight, srcUpLeft, copySize);
	}

	fftw_free(buffer);
	fftw_destroy_plan(plan);

	return result;
}
