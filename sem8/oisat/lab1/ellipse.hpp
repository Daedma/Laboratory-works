#pragma once

#include <stdexcept>

#include "sphere.hpp"

class ellipse : public sphere
{
public:
	ellipse(double a, double b, double c) :
		sphere(a)
	{
		scale({ 1, b / a, c / a });
	}
};