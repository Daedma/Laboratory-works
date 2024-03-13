// ЛР 1: Одномерные методы: 1) Дихотомия 2) Золотое сечение 3) Фибоначчи

// #include <limits>
#include <iostream>
#include <functional>
#include <cmath>
#include <vector>
#include <utility>

namespace
{
	constexpr double DEFAULT_ACCURACY = 1.e-6;
	constexpr size_t MAX_ITERATIONS = 1703;
}

double dihotomia(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	double middle;
	for (size_t i = 0; i != iterMax && (right - left) >= eps; ++i)
	{
		middle = (right + left) * .5;
		if (func(middle - eps) > func(middle + eps))
		{
			left = middle;
		}
		else
		{
			right = middle;
		}
	}
	return middle;
}

double goldenRatioDihotomia(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	double dx = right - left;
	double x1 = right - dx * REVERSE_PHI, x2 = left + dx * REVERSE_PHI;
	double fx1 = func(x1);
	double fx2 = func(x2);

	for (size_t i = 0; i != iterMax && (x2 - x1) >= eps; ++i)
	{
		if (fx1 >= fx2)
		{
			left = x1;
			dx = right - left;
			x1 = x2;
			x2 = left + dx * REVERSE_PHI;
			fx1 = fx2;
			fx2 = func(x2);
		}
		else
		{
			right = x2;
			dx = right - left;
			x2 = x1;
			x1 = right - dx * REVERSE_PHI;
			fx2 = fx1;
			fx1 = func(x1);
		}
	}
	return (x2 + x1) * 0.5;
}

std::pair<size_t, size_t> getFibonacciPairAbove(double val)
{
	size_t cur = 1, prev = 0;
	while (cur <= val)
	{
		cur += prev;
		prev = cur - prev;
	}
	return { cur, prev };
}

double fibonacciMethod(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY)
{
	double x1 = left, x2 = right, dx;
	auto [fn, fnm1] = getFibonacciPairAbove((right - left) / eps);
	while (fn != fnm1 && (x2 - x1) > eps)
	{
		dx = right - left;
		size_t fnm2 = fn - fnm1;
		x1 = left + (fnm2 * dx) / fn;
		x2 = left + (fnm1 * dx) / fn;
		fn = fnm1;
		fnm1 = fnm2;
		if (func(x1) < func(x2))
		{
			right = x2;
		}
		else
		{
			left = x1;
		}
	}
	return (x1 + x2) * .5;
}

int main()
{
	auto f = [](double x) {return -(x - 3) * pow(x + 6, 2) + 3;}; // -6
	std::cout << dihotomia(f, -8, -2) << '\n';
	std::cout << goldenRatioDihotomia(f, -8, -2) << '\n';
	std::cout << fibonacciMethod(f, -8, -2) << '\n';
}