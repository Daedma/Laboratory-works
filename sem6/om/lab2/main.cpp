// Многомерные методы :

// ЛР 2 :
// 	1) Определение типа vec_n(c++) / Vector(c#) и вспомогательных функций и операторов к нему.
// 	2) Дихотомия
// 	3) Золотое сечение
// 	4) Фибоначчи
// 	5) По - координатный спуск

#include <iostream>
#include <functional>
#include <cmath>
#include <utility>
#include "Mathter/Vector.hpp"

namespace
{
	constexpr double DEFAULT_ACCURACY = 1.e-6;
	constexpr size_t MAX_ITERATIONS = 1703;
}

template<int N>
using Vector = mathter::Vector<double, N>;

template<int N>
std::ostream& operator<<(std::ostream& os, const Vector<N>& rhs)
{
	os << '(';
	if (rhs.Dimension())
	{
		for (size_t i = 0; i != rhs.Dimension() - 1; i++)
		{
			os << rhs[i] << ';';
		}
		os << rhs[rhs.Dimension() - 1];
	}
	return os << ')';
}

template <typename T>
inline int sgn(T val)
{
	return (T(0) <= val) - (val < T(0));
}

template<int N, typename Func>
Vector<N> dihotomia(Func func, Vector<N> left, Vector<N> right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	Vector<N> middle, direction;
	for (size_t i = 0; i != iterMax && mathter::LengthSquared(right - left) >= eps * eps; ++i)
	{
		middle = (right + left) * .5;
		direction = mathter::Normalize(right - left) * eps;
		if (func(middle - direction) > func(middle + direction))
		{
			left = middle;
		}
		else
		{
			right = middle;
		}
	}
	return (left + right) * .5;
}

template<int N, typename Func>
Vector<N> goldenRatioDihotomia(Func func, Vector<N> left, Vector<N> right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	Vector<N> dx = right - left;
	Vector<N> x1 = right - dx * REVERSE_PHI, x2 = left + dx * REVERSE_PHI;
	double fx1 = func(x1);
	double fx2 = func(x2);

	eps *= 4. * eps;
	for (size_t i = 0; i != iterMax && mathter::LengthSquared(x2 - x1) >= eps; ++i)
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

std::pair<size_t, size_t> getFibonacciPairAbove(double val, size_t& outIterations)
{
	size_t cur = 1, prev = 0;
	while (cur <= val)
	{
		cur += prev;
		prev = cur - prev;
		++outIterations;
	}
	return { cur, prev };
}

template<int N, typename Func>
Vector<N> fibonacciMethod(Func func, Vector<N> left, Vector<N> right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	size_t it = 0;
	auto [fn, fnm1] = getFibonacciPairAbove(mathter::Distance(right, left) / eps, it);
	if (it > iterMax)
	{
		return (Vector<N>(NAN));
	}
	size_t fnm2 = fn - fnm1;
	Vector<N> dx = right - left;
	Vector<N> x1 = left + (fnm2 * dx) / fn;
	Vector<N> x2 = left + (fnm1 * dx) / fn;
	double fx1 = func(x1);
	double fx2 = func(x2);
	fn = fnm1;
	fnm1 = fnm2;
	fnm2 = fn - fnm1;
	while (it-- != 0)
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
	return (x1 + x2) * .5;
}

template<int N, typename Func>
Vector<N> perCoordDescend(Func func, Vector<N> start, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	Vector<N> ort(0.);
	Vector<N> cur;
	for (size_t i = 0; i != iterMax; i++)
	{
		ort[i % N] = 1.;
		int sign = sgn(func(start - ort * eps) - func(start + ort * eps));
		cur = goldenRatioDihotomia(func,
			start + sign * ort, start, eps, iterMax);
		if (mathter::LengthSquared(cur - start) < eps * eps)
		{
			return cur;
		}
		start = cur;
		ort[i % N] = 0.;
	}
	return cur;

}

static double test_func_2(const Vector<2>& x)
{
	return (x[0] - 5) * x[0] + (x[1] - 3) * x[1]; // min at point x = 2.5, y = 1.5
}

int main(int argc, char const* argv[])
{
	std::cout << "\n////////////////////\n";
	std::cout << "/// Lab. work #2 ///\n";
	std::cout << "////////////////////\n\n";

	Vector<2> x_0 = { 0, 0 };
	Vector<2> x_1 = { 5, 3 };

	std::cout << "{ x, y } = agrmin((x - 2) * (x - 2) + (y - 2) * (y - 2))\n";
	std::cout << "x_0 = " << x_0 << ", x_1 = " << x_1 << "\n";
	///  Для реализации по-координтаного спуска необходимо реализовать один из следующих трех методов для работы с vec_n
	std::cout << "bisect                : " << dihotomia(test_func_2, x_1, x_0) << "\n";
	std::cout << "golden_ratio          : " << goldenRatioDihotomia(test_func_2, x_1, x_0) << "\n";
	std::cout << "fibonacci             : " << fibonacciMethod(test_func_2, x_1, x_0) << "\n";
	std::cout << "\n";

	Vector<2> x_start = { -14, -33.98 };
	std::cout << "x_start = " << x_start << "\n";
	std::cout << "per_coord_descend     : " << perCoordDescend(test_func_2, x_start) << "\n";
	return 0;
}
