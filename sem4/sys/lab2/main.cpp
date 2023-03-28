#include <iostream>

int main()
{
	int result;
	int a = 1; // инициализируйте значения a, c, d заранее
	int c = 1;
	int d = 1231242246;
	int overf = 0, zerrof = 0;

	__asm__ volatile (
		"movl %[c], %%eax\n\t"
		"movl $4, %%ecx\n\t"
		"cltd\n\t"
		"idivl %%ecx\n\t"
		"imull $28, %[d], %%ecx\n\t"
		"jo error_of\n\t"
		"addl %%ecx, %%eax\n\t"
		"jo error_of\n\t"
		"movl %%eax, %%ebx\n\t"
		"jo error_of\n\t"
		"movl %[a], %%eax\n\t"
		"cmpl $0l, %[d]\n\t"
		"jz error_zf\n\t"
		"cltd\n\t"
		"idivl %[d]\n\t"
		"movl %%eax, %%ecx\n\t"
		"movl %%ebx, %%eax\n\t"
		"subl %[c], %%ecx\n\t"
		"jo error_of\n\t"
		"subl $1, %%ecx\n\t"
		"jz error_zf\n\t"
		"jo error_of\n\t"
		"cltd\n\t"
		"idivl %%ecx\n\t"
		"movl %%eax, %[r]\n\t"
		"jmp exit\n\t"
		"error_of:\n\t"
		"movl $1l, %[overf]\n\t"
		"jmp exit\n\t"
		"error_zf:\n\t"
		"movl $1l, %[zerrof]\n\t"
		"exit:\n\t"
		: [r] "=r" (result), [overf] "=r" (overf), [zerrof] "=r" (zerrof)
		: [a] "r" (a), [c] "r" (c), [d] "r" (d)
		: "%eax", "%ecx", "%ebx", "%rax"
		);
	if (zerrof)
	{
		std::cerr << "Division by zero!\n";
	}
	else if (overf)
	{
		std::cerr << "Overflow!\n";
	}
	else
	{
		std::cout << result << std::endl;
		std::cout << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
	}
}