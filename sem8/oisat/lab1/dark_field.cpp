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


int main()
{
	try
	{
		LightField field{ sizeX, sizeY, resX, resY };
		field.populate(image);

		LightField spectrum = field.fft();

		spectrum.applyFilter(filter);

		LightField output = spectrum.ifft();

		matplot::subplot(3, 1, 0);
		matplot::imagesc(field.abs());
		matplot::colorbar();

		matplot::subplot(3, 1, 1);
		matplot::imagesc(spectrum.abs());
		matplot::colorbar();

		matplot::subplot(3, 1, 2);
		matplot::imagesc(output.abs());
		matplot::colorbar();

		matplot::show();

		// plotLightField(field);
		// plotLightField(spectrum);
		// plotLightField(spectrum);
		// printLightField(field);
		// printLightField(spectrum);
		// printLightField(output);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}