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
	constexpr double minFreq = 0.1;
	if (std::abs(x) > minFreq && std::abs(y) > minFreq)
	{
		return 1.;
	}
	else
	{
		return 0.;
	}
};

inline constexpr auto image = bacteries;

int main()
{
    LightField field{ size, res };
    field.populate(image);

    LightField spectrum = field.fft();

    spectrum.applyFilter(filter);

    LightField output = spectrum.ifft();

    matplot::figure();
    matplot::imagesc(field.angle());
    matplot::colorbar();
    matplot::title("Original Field Angle");

    matplot::figure();
    matplot::imagesc(spectrum.abs());
    matplot::colorbar();
    matplot::title("Spectrum Magnitude");

    matplot::figure();
    matplot::imagesc(output.angle());
    matplot::colorbar();
    matplot::title("Output Field Angle");

	matplot::show();

	return 0;
}