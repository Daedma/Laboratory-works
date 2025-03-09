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
	double sizeX;

	double sizeY;

	size_t resX;

	size_t resY;

	std::vector<std::complex<double>, FFTWAllocator<std::complex<double>>> field;
public:
	LightField(double sizeX_, double sizeY_, size_t resX_, size_t resY_) :
		sizeX(sizeX_),
		sizeY(sizeY_),
		resX(resX_),
		resY(resY_),
		field(resX_* resY_)
	{}

	LightField& populate(std::function<std::complex<double>(double, double)> generator)
	{
		double hx = sizeX / resX;
		double hy = sizeY / resY;

		for (size_t i = 0; i != resX; ++i)
		{
			for (size_t j = 0; j != resY; ++j)
			{
				double x = (i - resX * 0.5) * hx;
				double y = (j - resY * 0.5) * hy;
				field[j * resX + i] = generator(x, y);
			}
		}

		return *this;
	}

	LightField& applyFilter(std::function<std::complex<double>(double, double)> filter)
	{
		double hx = sizeX / resX;
		double hy = sizeY / resY;

		for (size_t i = 0; i != resX; ++i)
		{
			for (size_t j = 0; j != resY; ++j)
			{
				double x = (i - resX * 0.5) * hx;
				double y = (j - resY * 0.5) * hy;
				field[j * resX + i] *= filter(x, y);
			}
		}

		return *this;
	}

	std::complex<double>& operator()(size_t x, size_t y) noexcept
	{
		return field[y * resX + x];
	}

	const std::complex<double>& operator()(size_t x, size_t y) const noexcept
	{
		return field[y * resX + x];
	}

	double getSizeX() const noexcept { return sizeX; }

	double getSizeY() const noexcept { return sizeY; }

	size_t getResolutionX() const noexcept { return resX; }

	size_t getResolutionY() const noexcept { return resY; }

	LightField fft(size_t newResX, size_t newResY) const { return fft_Impl(newResX, newResY, FFTW_FORWARD); }

	LightField fft() const { return fft(resX, resY); }

	LightField ifft(size_t newResX, size_t newResY) const { return fft_Impl(newResX, newResY, FFTW_BACKWARD); }

	LightField ifft() const { return ifft(resX, resY); }

	std::vector<std::vector<double>> abs() const
	{
		std::vector<std::vector<double>> result(resY, std::vector<double>(resX));
		for (size_t i = 0; i != resX; ++i)
		{
			for (size_t j = 0; j != resY; ++j)
			{
				result[j][i] = std::abs(field[j * resX + i]);
			}
		}
		return result;
	}

	std::vector<std::vector<double>> angle() const
	{
		std::vector<std::vector<double>> result(resY, std::vector<double>(resX));
		for (size_t i = 0; i != resX; ++i)
		{
			for (size_t j = 0; j != resY; ++j)
			{
				result[j][i] = std::arg(field[j * resX + i]);
			}
		}
		return result;
	}

private:
	LightField fft_Impl(size_t newResX, size_t newResY, int dftDirection) const;

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
		return reinterpret_cast<fftw_complex*>(&field[y * resX + x]);
	}

	const fftw_complex* get(size_t x, size_t y) const noexcept
	{
		return reinterpret_cast<const fftw_complex*>(&field[y * resX + x]);
	}

};