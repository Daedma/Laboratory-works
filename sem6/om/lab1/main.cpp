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
	size_t i = 0;
	for (; i != iterMax && (right - left) >= eps * 2; ++i)
	{
		if (func(middle - eps) > func(middle + eps))
		{
			left = (right + left) * .5;
		}
		else
		{
			right = (right + left) * .5;
		}
	}
	std::cout << "dihotomia::probes:" << i * 2 << '\n';
	std::cout << "dihotomia::range  :"<< right - left << '\n';
	return (right + left) * .5;
}

double goldenRatio(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	double dx = right - left;
	double x1 = right - dx * REVERSE_PHI, x2 = left + dx * REVERSE_PHI;
	double fx1 = func(x1);
	double fx2 = func(x2);
	size_t i = 0;

	for (; i != iterMax && (right - left) >= eps * 2; ++i)
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
	std::cout << "goldenRatio::probes:" << i + 2 << '\n';
	std::cout << "goldenRatio::range  :" << rhs - lhs << '\n';
	return (x2 + x1) * 0.5;
}

std::pair<size_t, size_t> getFibonacciPairAbove(double val, size_t & i)
{
	size_t cur = 1, prev = 0;
	while (cur <= val)
	{
		i++;
		cur += prev;
		prev = cur - prev;
	}
	return { cur, prev };
}

double fibonacci(std::function<double(double)> func, double left, double right, double eps = DEFAULT_ACCURACY)
{
	size_t i = 0;
	auto [fn, fnm1] = getFibonacciPairAbove((right - left) / eps, i);
	size_t fnm2 = fn - fnm1;
	double dx = right - left;
	double x1 = left + (fnm2 * dx) / fn;
	double x2 = left + (fnm1 * dx) / fn;
	double fx1 = func(x1);
	double fx2 = func(x2);
	fn = fnm1;
	fnm1 = fnm2;
	fnm2 = fn - fnm1;
	std::cout << "fibonacci::probes:" << i + 2 << '\n';
	while (--i)
	{
		if (fx1 < fx2)
		{
			right = x2;
			dx = right - left;
			x2 = x1;
			x1 = left + (fnm2 * dx) / fn;
			fx2 = fx1;
			fx1 = func(x1);
		}
		else
		{
			left = x1;
			dx = right - left;
			x1 = x2;
			x2 = left + (fnm1 * dx) / fn;
			fx1 = fx2;
			fx2 = func(x2);
		}
		fn = fnm1;
		fnm1 = fnm2;
		fnm2 = fn - fnm1;
	}
	std::cout << "fibonacci::range  :" << rhs - lhs << '\n';
	return (x1 + x2) * .5;
}

int main()
{
	auto f = [](double x) {return -(x - 3) * pow(x + 6, 2) + 3;}; // -6
	std::cout << dihotomia(f, -8, -2) << '\n';
	std::cout << goldenRatio(f, -8, -2) << '\n';
	std::cout << fibonacci(f, -8, -2) << '\n';
}
/*int main()
{
	auto f = [](double x) {return x * (x - 5); }; // 2.5
	std::cout << dihotomia  (f, -5.0, 5.0) << '\n';
	std::cout << goldenRatio(f, -5.0, 5.0) << '\n';
	std::cout << fibonacci  (f, -5.0, 5.0, 2e-6) << '\n';
}*/
