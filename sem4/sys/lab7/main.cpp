#include <iostream>
#include <cmath>

double ch(double x, double eps, double value)
{
	double result = 1., last = -1;
	size_t counter = 1;
	const double two = 2.;
	eps *= 1.5;
	asm volatile(
		"finit;"
		"fldl %[eps];"            // {eps}
		"fld1;"                   // {1, eps}

		"begin_loop:"             // {C_(k-1), eps}

		"fldl %[x];"              // {x, C_(k-1), eps}
		"fmul %%st;"              // {x*x, C_(k-1), eps}
		"fildl %[counter];"       // {k, x*x, C_(k-1), eps}
		"fmull %[two];"           // {2k, x*x, C_(k-1), eps}
		"fld1;"                   // {1, 2k, x*x, C_(k-1), eps}
		"fsub;"                   // {1 - 2k, 2k, x*x, C_(k-1), eps}
		"fchs;"                   // {2k - 1, 2k, x*x, C_(k-1), eps}
		"fmulp %%st(0), %%st(1);" // {2k(2k - 1), x*x, C_(k-1), eps}
		"fdivp %%st(0), %%st(1);" // {x*x/(2k(2k - 1)), C_(k-1), eps}
		"fmulp %%st(0), %%st(1);" // {x*x/(2k(2k - 1)) * C_(k-1), eps}

		"fucomi;"
		"jbe end_loop;"
		"fldl %[result];"
		"fadd;"
		"fstpl %[result];"
		"incq %[counter];"

		"jmp begin_loop;"
		"end_loop:"
		"fstl %[last];"

		: [result] "=m" (result), [counter] "+m" (counter), [last] "=m" (last)
		: [x] "m" (x), [eps] "m" (eps), [two] "m" (two)
		: "memory"
		);
	std::cout << counter << ' ' << last << ' ';
	return result;
}

double ch_cpp(double x, double eps)
{
	double result = 1;
	size_t k = 1;
	double last = 1;
	while (true)
	{
		double cur = x * x / ((2 * k - 1) * (2 * k)) * last;
		std::cout << cur << ' ';
		if (cur <= eps * 1.5)
		{
			return result;
		}
		else
		{
			last = cur;
			result += cur;
		}
		++k;
	}
	return result;
}

int main()
{
	double arg = 0.7;
	double precision = 1.e-5;
	std::cout << "asm : " << ch(arg, precision, 0) << std::endl;
	std::cout << "cpp : " << ch_cpp(arg, precision) << std::endl;
	std::cout << "std : " << std::cosh(arg) << std::endl;
}