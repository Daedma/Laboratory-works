// ЛР 1: Одномерные методы: 1) Дихотомия 2) Золотое сечение 3) Фибоначчи

#include <iostream>
#include <functional>
#include <limits>
#include <cmath>
#include <vector>
#include <utility>

namespace
{
	constexpr double DEFAULT_ACCURACY = 1.e-6;
}

double dihotomia(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY, size_t iterMax = std::numeric_limits<size_t>::max())
{
	double x1, x2, middle;
	for (size_t i = 0; i != iterMax && (x2 - x1) >= eps; ++i)
	{
		middle = (right + left) * .5;
		x1 = middle - eps;
		x2 = middle + eps;
		if (func(x1) > func(x2))
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

double goldenRatioDihotomia(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY, size_t iterMax = std::numeric_limits<size_t>::max())
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	double x1 = left, x2 = right, dx;
	for (size_t i = 0; i != iterMax && (x2 - x1) >= eps; ++i)
	{
		dx = right - left;
		x1 = right - dx * REVERSE_PHI;
		x2 = left + dx * REVERSE_PHI;
		if (func(x1) >= func(x2))
		{
			left = x1;
		}
		else
		{
			right = x2;
		}
	}
	return (x2 + x1) * 0.5;
}

size_t fibonacciNumber(size_t n)
{
	static std::vector<size_t> cash{0, 1};
	if (n <= cash.size())
	{
		return cash[n - 1];
	}
	return fibonacciNumber(n - 1) + fibonacciNumber(n - 2);
}

std::pair<size_t, size_t> getFibonacciAbove(double val)
{
	size_t i = 1;
	while (fibonacciNumber(i) <= val) { ++i; };
	return { fibonacciNumber(i), fibonacciNumber(i - 1) };
}

double fibonacciMethod(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY)
{
	double x1 = left, x2 = right, dx = right - left;
	auto [fn, fnm1] = getFibonacciAbove((right - left) / eps);
	while (fn != fnm1 && (x2 - x1) < eps)
	{
		dx = right - left;
		size_t fnm2 = fn - fnm1;
		x1 = left + static_cast<double>(fnm1) / fn * dx;
		x2 = left + static_cast<double>(fnm2) / fn * dx;
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
	std::cout << dihotomia([](double x) {return x * x - 3;}, -1, 4) << '\n';
	std::cout << goldenRatioDihotomia([](double x) {return x * x * x * x + 4;}, -3, 2) << '\n';
	std::cout << fibonacciMethod([](double x) {return x * x - 4;}, -1, 2) << '\n';
}