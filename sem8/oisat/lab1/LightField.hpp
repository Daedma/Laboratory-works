#pragma once

#include <vector>
#include <complex>
#include <functional>
#include <utility>

#include "FFTWAllocator.hpp"
#include "MatrixProxy.hpp"

class LightField
{
private:
	double size;

	size_t res;

	std::vector<std::complex<double>, FFTWAllocator<std::complex<double>>> field;
public:
	LightField(double size_, size_t res_) :
		size(size_),
		res(res_),
		field(res_* res_)
	{}

	LightField& populate(std::function<std::complex<double>(double, double)> generator)
	{
		double h = size / res;

		for (size_t i = 0; i != res; ++i)
		{
			for (size_t j = 0; j != res; ++j)
			{
				double x = (i - res * 0.5) * h;
				double y = (j - res * 0.5) * h;
				field[j * res + i] = generator(x, y);
			}
		}

		return *this;
	}

	LightField& applyFilter(std::function<std::complex<double>(double, double)> filter)
	{
		double h = size / res;

		for (size_t i = 0; i != res; ++i)
		{
			for (size_t j = 0; j != res; ++j)
			{
				double x = (i - res * 0.5) * h;
				double y = (j - res * 0.5) * h;
				field[j * res + i] *= filter(x, y);
			}
		}

		return *this;
	}

	std::complex<double>& operator()(size_t x, size_t y) noexcept
	{
		return field[y * res + x];
	}

	const std::complex<double>& operator()(size_t x, size_t y) const noexcept
	{
		return field[y * res + x];
	}

	double getSize() const noexcept { return size; }

	size_t getResolution() const noexcept { return res; }

	LightField fft() const { return fft_Impl(FFTW_FORWARD); }

	LightField ifft() const { return fft_Impl(FFTW_BACKWARD); }

	// Функция для вычисления абсолютных значений
	std::vector<std::vector<double>> abs(size_t output_size) const
	{
		return processMatrix(output_size, [](const std::complex<double>& value) {
			return std::abs(value); // Модуль комплексного числа
			});
	}

	// Функция для вычисления углов (фазы)
	std::vector<std::vector<double>> angle(size_t output_size) const
	{
		return processMatrix(output_size, [](const std::complex<double>& value) {
			return std::arg(value); // Фаза комплексного числа
			});
	}

private:
	LightField fft_Impl(int dftDirection) const;

	fftw_complex* get() noexcept
	{
		return reinterpret_cast<fftw_complex*>(field.data());
	}

	const fftw_complex* get() const noexcept
	{
		return reinterpret_cast<const fftw_complex*>(field.data());
	}

	fftw_complex* get(size_t x, size_t y) noexcept
	{
		return reinterpret_cast<fftw_complex*>(&field[y * res + x]);
	}

	const fftw_complex* get(size_t x, size_t y) const noexcept
	{
		return reinterpret_cast<const fftw_complex*>(&field[y * res + x]);
	}

	void shift(const fftw_complex* src, size_t srcSize,
		size_t srcStep, fftw_complex* dst, size_t dstSize, size_t dstStep) const;

	void fit(const fftw_complex* src, size_t srcSize,
		size_t srcStep, fftw_complex* dst, size_t dstSize, size_t dstStep) const;

	void transpose(fftw_complex* src, size_t srcSize, size_t srcStep) const;

	std::vector<std::vector<double>> processMatrix(
		size_t output_size,
		const std::function<double(const std::complex<double>&)>& operation // Функция, применяемая к элементам
	) const
	{
		if (output_size == 0)
		{
			throw std::invalid_argument("Output size must be greater than 0");
		}

		std::vector<std::vector<double>> result(output_size, std::vector<double>(output_size));
		double scale_x = static_cast<double>(res - 1) / (output_size - 1);
		double scale_y = static_cast<double>(res - 1) / (output_size - 1);

		for (size_t i = 0; i < output_size; ++i)
		{
			for (size_t j = 0; j < output_size; ++j)
			{
				double original_x = j * scale_x;
				double original_y = i * scale_y;

				size_t x1 = static_cast<size_t>(original_x);
				size_t y1 = static_cast<size_t>(original_y);
				size_t x2 = (x1 + 1 < res) ? x1 + 1 : x1;
				size_t y2 = (y1 + 1 < res) ? y1 + 1 : y1;

				double q11 = operation(field[y1 * res + x1]);
				double q12 = operation(field[y2 * res + x1]);
				double q21 = operation(field[y1 * res + x2]);
				double q22 = operation(field[y2 * res + x2]);

				double x = original_x - x1;
				double y = original_y - y1;

				result[j][i] = bilinearInterpolation(q11, q12, q21, q22, x, y);
			}
		}
		return result;
	}

	static double bilinearInterpolation(double q11, double q12, double q21, double q22, double x, double y)
	{
		double r1 = q11 * (1 - x) + q21 * x;
		double r2 = q12 * (1 - x) + q22 * x;
		return r1 * (1 - y) + r2 * y;
	}
};