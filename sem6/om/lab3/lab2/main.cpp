// Многомерные методы :

// ЛР 3:
// 6) Градиентный спуск спуск
// 7) метод сопряжённых градиентов

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
Vector<N> goldenRatio(Func func, Vector<N> left, Vector<N> right, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	Vector<N> dx = right - left;
	Vector<N> x1 = right - dx * REVERSE_PHI, x2 = left + dx * REVERSE_PHI;
	double fx1 = func(x1);
	double fx2 = func(x2);
	size_t i = 0;
	eps *= 4. * eps;
	for (; i != iterMax && mathter::LengthSquared(x2 - x1) >= eps; ++i)
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
	Vector<2> x_start = { -14, -33.98 };
	std::cout << "x_start = " << x_start << "\n";
	return 0;
}
