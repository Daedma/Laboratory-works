#include <iostream>

int f_cpp(int a, int b)
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
	int a, b;
	std::cin >> a >> b;
	int result;
	int error = 0;
	asm volatile(
		"xorl %[error], %[error];"
		"movl	%[a], %%eax;"            // eax: a
		"movl	%[b], %%esi;"            // esi: b
		"movl %%eax, %%ecx;"             // ecx: a  
		"subl %%esi, %%ecx;"             // ecx: a - b
		"jo error_of;"
		"jle less_equal;"                // to a <= b
		"addl %%eax, %%esi;"             // esi: a + b
		"jo error_of;"
		"jz error_zf;"
		"movl %%ecx, %%eax;"             // eax: a - b
		"jmp division;"                  // to end
		"less_equal:;"                   // if a <= b
		"jne less;"                      // to a < b
										 // if a == b
		"imull %%eax, %%eax;"            // eax: a^2
		"jo error_of;"
		"negl %%eax;"                    // eax: -(a^2)
		"jmp end;"                       // to exit
		"less:;"                         // if a < b
		"addl $0, %%esi;"				 // check zero
		"jz error_zf;"
		"leal (%%rax, %%rax, 2), %%eax;" //eax: a + 2 * a
		"addl	$-2, %%eax;"             //eax: 3 * a - 2
		"jo error_of;"
		"division:;"
		"cltd;"
		"idivl %%esi;"                   // eax: (a - b) / (a + b) or (3 * a - 2) / b
		"end:;"
		"movl %%eax, %[result];"
		"jmp exit;"
		"error_zf:;"
		"movl $1, %[error];"
		"jmp exit;"
		"error_of:;"
		"movl $2, %[error];"
		"exit:;"
		: [result] "=r" (result), [error] "=r" (error)
		: [a] "r" (a), [b] "r" (b)
		: "eax", "esi", "ecx", "rax"
		);
	if (error == 1)
	{
		std::cerr << "Division by zero.\n";
		return EXIT_FAILURE;
	}
	if (error == 2)
	{
		std::cerr << "Overflow.\n";
		return EXIT_FAILURE;
	}
	std::cout << result << std::endl;
	std::cout << f_cpp(a, b) << std::endl;
}