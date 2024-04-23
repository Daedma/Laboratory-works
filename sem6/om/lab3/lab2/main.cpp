// Многомерные методы :

// ЛР 3:
// 6) Градиентный спуск спуск
// 7) метод сопряжённых градиентов

// ЛР 4:
// 8) Определение типа mat_mn(c++) / Matrix(c#) и вспомогательных функций и операторов к нему.
// 9) Метод Ньютона - Рафсона.
// 10)Функции внешнего и внутреннего штрафа.

#include <iostream>
#include <functional>
#include <cmath>
#include <utility>
#include "Mathter/Vector.hpp"
#include "Mathter/Matrix.hpp"
#include "Mathter/IoStream.hpp" 

namespace
{
	constexpr double DEFAULT_ACCURACY = 1.e-6;
	constexpr size_t MAX_ITERATIONS = 1703;
}

template<int N>
using Vector = mathter::Vector<double, N>;

template<int Rows, int Columns>
using Matrix = mathter::Matrix<double, Rows, Columns, mathter::eMatrixOrder::PRECEDE_VECTOR>;

template <typename T>
inline int sgn(T val)
{
	return (T(0) <= val) - (val < T(0));
}


template<int N, typename Func>
double partialDerivative(Func func, uint32_t nvar, const Vector<N>& x, double eps)
{
	Vector<N> xipe = x;
	xipe[nvar] += eps;
	return (func(xipe) - func(x)) / eps;
}

template<int N, typename Func>
Vector<N> gradient(Func func, const Vector<N>& x, double eps)
{
	Vector<N> result;
	for (int i = 0; i != N; ++i)
	{
		result[i] = partialDerivative(func, i, x, eps);
	}
	return result;
}

template<int N, typename Func>
double partialDerivative2(Func func, uint32_t nvar1, uint32_t nvar2, const Vector<N>& x, double eps)
{
	Vector<N> xipe = x;
	xipe[nvar1] += eps;
	return (partialDerivative(func, nvar2, xipe, eps * 0.5) - partialDerivative(func, nvar2, x, eps * 0.5)) / eps;
}


template<int N, typename Func>
Matrix<N, N> hessian(Func func, const Vector<N> x, double eps)
{
	Matrix<N, N> result;
	for (int i = 0; i != N; ++i)
	{
		for (int j = i; j != N; ++j)
		{
			result(i, j) = result(j, i) = partialDerivative2(func, i, j, x, eps);
		}
	}
	return result;
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

template<int N, typename Func>
Vector<N> gradientDescent(Func func, Vector<N> start, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	Vector<N> cur;
	for (size_t i = 0; i != iterMax; i++)
	{
		cur = goldenRatio(func,
			start - gradient(func, start, eps), start, eps, iterMax);
		if (mathter::LengthSquared(cur - start) < eps * eps)
		{
			return cur;
		}
		start = cur;
	}
	return cur;
}

template<int N, typename Func>
Vector<N> conjGradientDescend(Func func, Vector<N> start, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	Vector<N> gradPrev = -gradient(func, start, eps);
	Vector<N> gradCur;
	Vector<N> cur;
	double omega;
	for (size_t i = 0; i != iterMax; ++i)
	{
		cur = start + gradPrev;
		if (mathter::LengthSquared(cur - start) < eps * eps)
		{
			return cur;
		}
		cur = goldenRatio(func, cur, start, eps, iterMax);
		gradCur = -gradient(func, cur, eps);
		// omega = mathter::LengthSquared(gradCur) / mathter::LengthSquared(gradPrev); // Fletcher–Reeves
		omega = std::max(0., mathter::Dot(gradCur, gradCur - gradPrev)) / mathter::LengthSquared(gradPrev); // Polak–Ribière
		gradPrev = gradPrev * omega + gradCur;
		start = cur;
	}
	return cur;
}

template<int N, typename Func>
Vector<N> newtoneRaphson(Func func, Vector<N> start, double eps = DEFAULT_ACCURACY, size_t iterMax = MAX_ITERATIONS)
{
	Vector<N> cur;
	Vector<N> grad;
	Matrix<N, N> hess;
	for (size_t i = 0; i != iterMax; ++i)
	{
		grad = gradient(func, start, eps);
		hess = mathter::Inverse(hessian(func, start, eps));
		cur = start - hess * grad;
		if (mathter::LengthSquared(cur - start) < eps * eps)
		{
			return cur;
		}
		start = cur;
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
	std::cout << "/// Lab. work #3 ///\n";
	std::cout << "////////////////////\n\n";
	Vector<2> x_start = { -14, -33.98 };
	std::cout << "x_start = " << x_start << "\n";
	std::cout << "gradient_descend      : " << gradientDescent(test_func_2, x_start) << "\n";
	std::cout << "conj_gradient_descend : " << conjGradientDescend(test_func_2, x_start) << "\n";

	std::cout << "\n////////////////////\n";
	std::cout << "/// Lab. work #4 ///\n";
	std::cout << "////////////////////\n\n";
	x_start = { -12.0, -15.0 };
	std::cout << "newtone_raphson       : " << newtoneRaphson(test_func_2, x_start) << "\n";
	return 0;
}
