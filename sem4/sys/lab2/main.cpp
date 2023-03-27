#include <iostream>

int main()
{
	int result;
	int a = 1; // инициализируйте значения a, c, d заранее
	int c = 1;
	int d = 1;

	asm(
		"movl %[c], %%eax;"
		"movl $4, %%ecx;"
		"cltd;"
		"idivl %%ecx;"
		"imull $28, %[d], %%ecx;"
		"addl %%ecx, %%eax;"
		"movl %%eax, %%ebx;"
		"movl %[a], %%eax;"
		"cltd;"
		"idivl %[d];"
		"movl %%eax, %%ecx;"
		"movl %%ebx, %%eax;"
		"subl %[c], %%ecx;"
		"subl $1, %%ecx;"
		"cltd;"
		"idivl %%ecx;"
		"movl %%eax, %[r];"
		: [r] "=r" (result)
		: [a] "r" (a), [c] "r" (c), [d] "r" (d)
		: "%eax", "%ecx", "%ebx", "%rax"
	);
	std::cout << result << std::endl;
	std::cout << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
}