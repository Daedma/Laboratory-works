#pragma once

#include <complex>
#include <cmath>

#include <matplot/matplot.h>

#include "LightField.hpp"

namespace
{
	constexpr size_t res = 10;
	constexpr double size = 5. * 2.;

	constexpr auto bacteries = [](double x, double y) -> std::complex<double> {
		using namespace std::complex_literals;
		constexpr double pi = 3.14159265358979323846;
		constexpr double A = 8.;
		constexpr double B = 0.8;
		constexpr double C = 8.;
		return std::exp(1.i * B * pi * std::sin(A * x) * std::sin(C * y));
		};

	constexpr auto gauss = [](double x, double y) -> std::complex<double> {
		return std::exp(-x * x - y * y);
		};

	constexpr auto ones = [](double x, double y) -> std::complex<double> {
		return 1.;
		};
}
