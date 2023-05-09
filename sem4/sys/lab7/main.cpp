#include <iostream>
#include <cmath>
#include <limits>
#include <tuple>
#include <iomanip>

std::tuple<double, size_t, double> ch(double x, double eps, double value)
{
	double result = 1.;
	double error = -1.;
	size_t counter = 1;
	eps *= 1.5;                   // R_n < 2/3*v_n
	asm volatile(
		"finit;"
		"fldl %[eps];"            // {eps}
		"fld1;"                   // {1, eps}
		/// Начало цикла
		"begin_loop:"             // {C_(k-1), eps}

		"fildl %[counter];"       // {k, C_(k-1), eps}
		"fadd %%st;"              // {2k, C_(k-1), eps}
		"fld1;"                   // {1, 2k, C_(k-1), eps}
		"fsub %%st(1);"           // {1 - 2k, 2k, C_(k-1), eps}
		"fchs;"                   // {2k - 1, 2k, C_(k-1), eps}
		"fmulp %%st(0), %%st(1);" // {2k(2k - 1), C_(k-1), eps}
		"fldl %[x];"              // {x, 2k(2k - 1), C_(k-1), eps}
		"fmul %%st;"              // {x*x, 2k(2k - 1), C_(k-1), eps}
		"fdivp %%st(0), %%st(1);" // {x*x/(2k(2k - 1)), C_(k-1), eps}
		"fmulp %%st(0), %%st(1);" // {x*x/(2k(2k - 1)) * C_(k-1), eps}

		"fucomi;"                 // сравнение с eps
		"jbe end_loop;"           // если меньше, то заканчиваем цикл
		"fldl %[result];"         // {result, C_k, eps}
		"fadd %%st(1);"           // {result + C_k, C_k, eps}
		"fstpl %[result];"        // {C_k, eps}
		"incq %[counter];"        // инкрементируем счетчик

		"jmp begin_loop;"
		/// Конец цикла
		"end_loop:"               // вычисление достигнутой погрешности
		"fldl %[result];"         // {result, C_k, eps}
		"fsubl %[value];"          // {result - value, C_k, eps}
		"fabs;"                   // {|value - result|, C_k, eps}
		"fstl %[error];"          //

		: [result] "+m" (result), [counter] "+m" (counter), [error] "=m" (error)
		: [x] "m" (x), [eps] "m" (eps), [value] "m" (value)
		: "memory"
		);
	return { result, counter, error };
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
	double x, eps;
	std::cout << "Enter x\n>";
	std::cin >> x;
	std::cout << "Enter precision\n>";
	std::cin >> eps;
	auto [result, count, error] = ch(x, eps, cosh(x));
	constexpr auto max_precision{ std::numeric_limits<double>::digits10 + 1 };
	std::cout << "Result                                  : " << std::setprecision(max_precision) << result << '\n';
	std::cout << "Number of sumarized terms of the series : " << std::setprecision(max_precision) << count << '\n';
	std::cout << "Error                                   : " << std::setprecision(max_precision) << error << '\n';
}