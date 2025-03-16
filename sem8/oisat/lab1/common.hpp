#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <matplot/matplot.h>

#include "LightField.hpp"

namespace
{
	namespace imageParams
	{
		constexpr size_t res = 1000;
		constexpr double size = 5. * 2.;
	}

	namespace constants
	{
		constexpr double pi = 3.14159265358979323846;
		constexpr double twopi = 6.28318530718;
		constexpr double halfpi = 1.57079632679;
	};

	namespace fields
	{
		constexpr auto sinphase = [](double freqX, double freqY, double phaseFactor) {
			return [freqX, freqY, phaseFactor](double x, double y) -> std::complex<double> {
				using constants::pi;
				const double amp = 1.;
				const double phase = phaseFactor * pi * std::sin(freqX * x) * std::sin(freqY * y);
				return std::polar(amp, phase);
				};
			};

		constexpr auto sinamp = [](double freqX, double freqY, double ampFactor) {
			return [freqX, freqY, ampFactor](double x, double y) -> std::complex<double> {
				using constants::pi;
				const double amp = std::abs(ampFactor * pi * std::sin(freqX * x) * std::sin(freqY * y));
				const double phase = 1.;
				return std::polar(amp, phase);
				};
			};

		constexpr auto gauss = [](double x, double y) -> std::complex<double> {
			return std::exp(-x * x - y * y);
			};

		constexpr auto ones = [](double x, double y) -> std::complex<double> {
			return 1.;
			};

		constexpr auto bacteries = [](double width, double height, double stepX, double stepY, double amp, double phase) {
			using constants::pi;
			const double gridSize = std::max({ width, height, stepX, stepY });
			const std::random_device::result_type baseSeed = std::random_device{}();

			return [=](double x, double y) ->std::complex<double> {
				const int16_t i = static_cast<int16_t>(std::floor(x / gridSize));
				const int16_t j = static_cast<int16_t>(std::floor(y / gridSize));
				const double x0 = gridSize * (i + 0.5);
				const double y0 = gridSize * (j + 0.5);

				const std::random_device::result_type seed = baseSeed ^ (i << 16) ^ j;
				std::mt19937 gridGen(seed);
				std::uniform_real_distribution<double> phiDis(0., pi);
				const double phi = phiDis(gridGen);

				const double fitX = (x - x0) * std::cos(-phi) - (y - y0) * std::sin(-phi);
				const double fitY = (x - x0) * std::sin(-phi) + (y - y0) * std::cos(-phi);

				const bool isInside = (fitX * fitX / (width * width)) + (fitY * fitY / (height * height)) <= 0.25;

				return std::polar(amp * static_cast<double>(isInside), phase);
				};
			};
	}

	namespace filters
	{
		constexpr auto darkfield = [](double area) {
			return [area](double x, double y)->std::complex<double> {
				const bool isOutside = std::abs(x) > area || std::abs(y) > area;
				return { static_cast<double>(isOutside), 0. };
				};
			};

		constexpr auto zernick = [](double area) {
			return [area](double x, double y) -> std::complex<double> {
				const bool isOutside = std::abs(x) > area || std::abs(y) > area;
				if(isOutside)
				{
					return {1., 0.};
				}
				else
				{
					return {0., 1.};
				}
				};
			};

		constexpr auto derivative = [](int varPos) {
			return	[varPos](double x, double y) -> std::complex<double> {
				using constants::pi;
				using namespace std::complex_literals;
				const double var = varPos == 0 ? x : y;
				return 2.i * pi * var;
				};
			};

		constexpr auto identity = [](double x, double y) -> std::complex<double> {
			return 1.;
			};
	}

	namespace app
	{
		void plotField(LightField field, const std::string& label)
		{
			matplot::figure();

			matplot::subplot(1, 2, 0);
			matplot::imagesc(field.angle());
			matplot::colorbar();
			matplot::title("Phase of " + label);

			matplot::subplot(1, 2, 1);
			matplot::imagesc(field.abs());
			matplot::colorbar();
			matplot::title("Amplitude of " + label);
		}

		void demostrateFiltration(std::function<std::complex<double>(double, double)> field,
			std::function<std::complex<double>(double, double)> filter,
			double fieldSize = imageParams::size, size_t fieldRes = imageParams::res)
		{
			LightField inputField{ fieldSize, fieldRes };
			inputField.populate(field);

			LightField spectrum = inputField.fft();
			spectrum.applyFilter(filter);

			LightField outputField = spectrum.ifft();

			plotField(inputField, "Input Field");

			plotField(spectrum, "Spectrum after filter applying");

			plotField(outputField, "Output Field");

			matplot::show();
		}
	};
}
