#pragma once

#include <complex>
#include <cmath>

#include <matplot/matplot.h>

#include "LightField.hpp"

namespace
{
	constexpr size_t resX = 100;
	constexpr size_t resY = 100;
	constexpr double sizeX = 5. * 2.;
	constexpr double sizeY = 5. * 2.;

	constexpr double A = 8.;
	constexpr double B = 0.8;
	constexpr double C = 8.;
	constexpr auto image = [](double x, double y) -> std::complex<double> {
		using namespace std::complex_literals;
		constexpr double pi = 3.14159265358979323846;
		// return std::exp(1.i * (B * pi * std::sin(A * x) * std::sin(C * y)));
		return std::exp(-x * x - y * y);
		};
}

void plotLightField(const LightField& field)
{
	matplot::imagesc(field.abs());
	matplot::colorbar();
	matplot::show();
}

void printLightField(const LightField& field)
{
	auto abs = field.abs();
	for (size_t i = 0; i != resY; ++i)
	{
		for (size_t j = 0; j != resX; ++j)
		{
			std::cout << abs[i][j] << ' ';
		}
		std::cout << std::endl;
	}
}