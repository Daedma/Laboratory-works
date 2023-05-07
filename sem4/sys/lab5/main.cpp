#include <iostream>

int main()
{
	double a, d, c;
	std::cout << "a : ";
	std::cin >> a;
	std::cout << "c : ";
	std::cin >> c;
	std::cout << "d : ";
	std::cin >> d;
	double result = 0;
	double four = 4;
	double twei = 28;
	double one = 1;

	int error = 0;

	asm volatile(
		"finit;"              // {}
		"fldl %[c];"          // {c}
		"fdivl %[four];"      // {c/4}
		"fldl %[d];"          // {d, c/4}
		"fmull %[twei];"      // {28*d, c/4}
		"fadd;"               // {d*28 + c/4, c/4}

		"fldl %[a];"          // {a, d*28 + c/4, c/4}

		"fldz;"               // обработка деления на ноль
		"fldl %[d];"
		"fucompp;"
		"fnstsw %%ax;"
		"sahf;"
		"jz error_0;"

		"fdivl %[d];"         // {a/d, d*28 + c/4, c/4}

		"fsubl %[c];"         // {a/d - c, d*28 + c/4, c/4}
		"fsubl %[one];"       // {a/d - c - 1, d*28 + c/4, c/4}

		"fldz;"               // обработка деления на ноль
		"fucomp;"
		"fnstsw %%ax;"
		"sahf;"
		"jz error_0;"

		"fxch %%st(1);"       // {d*28 + c/4, a/d - c - 1, c/4}
		"fdiv;"               // {(c / 4 + 28 * d) / (a / d - c - 1), d*28 + c/4, c/4}
		"fstpl %[result];"    //
		"jmp exit;"

		"error_0:"
		"movl $1, %[err];"

		"exit:;"
		: [result] "=m" (result)
		: [a] "m" (a), [d] "m" (d), [c] "m" (c), [four] "m" (four), [twei] "m" (twei), [one] "m" (one), [err] "m" (error)
		: "memory", "ax"
		);
	if (error)
	{
		std::cout << "Division by zero\n";
	}
	else
	{
		std::cout << "asm : " << result << std::endl;
	}
	std::cout << "cpp : " << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
}