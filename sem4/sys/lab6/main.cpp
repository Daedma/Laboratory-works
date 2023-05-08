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

	int error = 0;
	short status = 0;

	asm volatile(
		"finit;"               // инициализация сопроцессора
		"fldl %[b];"           // {b}
		"fldl %[a];"           // {a, b}
		"fucomi;"			   // сравнение a и b
		"ja great;"
		"jb less;"
		// a == b
		"fmul;"                // {a*b, b}
		"fchs;"                // {-a*b, b}
		"jmp end;"
		// a > b
		"great:"
		"fadd;"                // {a + b, b}
		"fldz;"                // {0, a + b, b}
		"fucomip;"             // {a + b, b} # сравнение с (a + b) с нулем
		"je error_0;"
		"fldl %[a];"           // {a, a + b, b}
		"fsubl %[b];"          // {a - b, a + b, b}
		"fdiv;"                // {(a - b)/(a + b), a + b, b}
		"jmp end;"
		// a < b
		"less:"
		"fldz;"                // {0, a, b}
		"fucomip %%st(2);"     // {a, b} # сравнение b с нулем
		"je error_0;"
		"fmull %[c3];"         // {a * 3, b}
		"faddl %[cm2];"        // {a * 3 - 2, b}
		"fdiv;"                // {(a*3 - 2)/b, b}
		"jmp end;"
		// обработка ошибки деления на ноль
		"error_0:"
		"movl $1, %[error];"
		"jmp exit;"
		// запись результата
		"end:"
		"fstl %[result];"
		"exit:"
		"fxam;"
		"fstsw %%ax;"
		"and $0b0100010100000000, %%ax;"
		"mov %%ax, %[status];"

		: [result] "=m" (result), [error] "=m" (error), [status] "=m" (status)
		: [a] "m" (a), [b] "m" (b), [c3] "m" (c3), [cm2] "m" (cm2)
		: "memory", "ax"
		);

	if (error)
	{
		std::cout << "Division by zero.\n";
	}
	else
	{
		std::cout << "asm : " << result << std::endl;
		switch (status)
		{
		case 0x0:
			std::cout << "Unsupported format\n";
			break;
		case 0x100:
			std::cout << "NaN\n";
			break;
		case 0x400:
			std::cout << "Finite number\n";
			break;
		case 0x500:
			std::cout << "Infinity\n";
			break;
		case 0x4000:
			std::cout << "Zero\n";
			break;
		case 0x4100:
			std::cout << "Empty register\n";
			break;
		case 0x4400:
			std::cout << "An unnormalized number\n";
			break;
		default:
			break;
		}
	}
	std::cout << "cpp : " << f_cpp(a, b) << std::endl;
}