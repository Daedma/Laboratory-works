#include <iostream>

double f_cpp(double a, double b)
{
	if (a > b)
	{
		return (a - b) / (a + b);
	}
	else if (a == b)
	{
		return -a * b;
	}
	else if (a < b)
	{
		return (3 * a - 2) / b;
	}
	return 0;
}

int main()
{
	double a, b;
	std::cout << "Enter a\n>";
	std::cin >> a;
	std::cout << "Enter b\n>";
	std::cin >> b;

	double result = 0;
	const double c3 = 3.;
	const double cm2 = -2.;

	asm volatile(
		"finit;"
		"fldl %[b];"
		"fldl %[a];"
		"fucomi;"
		"ja great;"
		"jb less;"
		// a == b
		"fmul;"
		"fchs;"
		"jmp end;"
		// a > b
		"great:"
		"fadd;"
		"fldl %[a];"
		"fsubl %[b];"
		"fdiv;"
		"jmp end;"
		// a < b
		"less:"
		"fmull %[c3];"
		"faddl %[cm2];"
		"fdiv;"
		"jmp end;"
		// запись результата
		"end:"
		"fstl %[result];"

		: [result] "=m" (result)
		: [a] "m" (a), [b] "m" (b), [c3] "m" (c3), [cm2] "m" (cm2)
		: "memory"
		);

	std::cout << "asm : " << result << std::endl;
	std::cout << "cpp : " << f_cpp(a, b) << std::endl;
}