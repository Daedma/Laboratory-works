// ЛР 1: Одномерные методы: 1) Дихотомия 2) Золотое сечение 3) Фибоначчи

#include <iostream>
#include <functional>
#include <limits>

double dihotomia(std::function<double(double)> func, double left, double right, double eps, size_t iterMax = std::numeric_limits<size_t>::max())
{
	double x1, x2, halfDistance = INFINITY;
	for (size_t i = 0; i != iterMax && halfDistance > eps; ++i)
	{
		halfDistance = (right - left) * .5;
		x1 = halfDistance - eps;
		x2 = halfDistance + eps;
		if (func(x1) > func(x2))
		{
			left = x1;
		}
		else
		{
			right = x2;
		}
	}
	return x1 + halfDistance;
}

double goldenRatioDihotomia(std::function<double(double)> func, double left, double right, double eps, size_t iterMax = std::numeric_limits<size_t>::max())
{

}