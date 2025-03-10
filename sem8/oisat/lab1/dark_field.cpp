#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <stdexcept>

#include <fftw3.h>
#include <matplot/matplot.h>

#include "LightField.hpp"
#include "common.hpp"


inline constexpr auto filter = [](double x, double y)->std::complex<double> {
	return { 1., 0. };
	};

inline constexpr auto image = bacteries;

int main()
{
	LightField field{ size, res };
	field.populate(image);

	LightField spectrum = field.fft();

	spectrum.applyFilter(filter);

	LightField output = spectrum.ifft();

	matplot::subplot(1, 3, 0);
	matplot::imagesc(field.angle());
	matplot::colorbar();

	matplot::subplot(1, 3, 1);
	matplot::imagesc(spectrum.abs());
	matplot::colorbar();

	matplot::subplot(1, 3, 2);
	matplot::imagesc(output.angle());
	matplot::colorbar();

	matplot::show();

	return 0;
}