#pragma once

#include <vector>
#include <complex>
#include <functional>
#include <utility>

class LightField
{
private:
	double sizeX;

	double sizeY;

	size_t resX;

	size_t resY;

	std::vector<std::complex<double>> field;
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
				field[j * resX + i] = generator(i * hx, j * hy);
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
				field[j * resX + i] *= filter(i * hx, j * hy);
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

	LightField fft(size_t newResX, size_t newResY) const;

	LightField ifft() const;

};