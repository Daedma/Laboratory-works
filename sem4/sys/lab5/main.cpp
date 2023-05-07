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
	int64_t four = 4;
	int64_t twei = 28;


	asm volatile(
		"finit;"            // {}
		"fldl %[c];"         // {c}
		"fildq %[four];"     // {4, c}
		"fdivp;"            // {c/4}
		"fildq %[twei];"     // {28, c/4}
		"fldl %[d];"         // {d, 28, c/4}
		"fmulp;"             // {28 * d, c/4}
		"faddp;"            // {28 * d + c/4}
		"fldl %[a];"         // {a, 28 * d + c/4}
		"fldl %[d];"         // {d, a, 28 * d + c/4}
		"fdivp;"            // {a/d, 28 * d + c/4}
		"fldl %[c];"         // {c, a/d, 28 * d + c/4}
		"fsubp;"            // {a/d - c, 28 * d + c/4}
		"fld1;"             // {1, a/d - c, 28 * d + c/4}
		"fsubp;"            // {a/d - c - 1, 28 * d + c/4}
		"fdivp;"            // {(c / 4 + 28 * d) / (a / d - c - 1)}
		"fstpl %[result];"   // {}
		: [result] "=m" (result)
		: [a] "m"(a), [d]"m"(d), [c]"m"(c), [four] "m" (four), [twei] "m" (twei)
		: "memory"
		);
	std::cout << "asm : " << result << std::endl;
	std::cout << "cpp : " << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
}