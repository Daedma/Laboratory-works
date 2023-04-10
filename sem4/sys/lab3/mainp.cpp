#include <iostream>

void f(int a, int b)
{
	asm volatile(
		"movl	%[a], %%eax;"            // eax: a
		"movl	%[b], %%r8d;"            // r8d: b
		"movl %%eax, %%ecx;"             // ecx: a  
		"subl %%r8d, %%ecx;"             // ecx: a - b
		"jle less_equal;"                // to a <= b
		"addl %%eax, %%r8d;"             // r8d: a + b
		"movl %%ecx, %%eax;"             // eax: a - b
		"jmp division;"                  // to end
		"less_equal :;"                  // if a <= b
		"jne less;"                      // to a < b
										 // if a == b
		"imull %%eax, %%eax;"            // eax: a^2
		"negl %%eax;"                    // eax: -(a^2)
		"jmp end;"                       // to exit
		"less :;"                        // if a < b
		"leal (%%rax, %%rax, 2), %%eax;" //eax: a + 2 * a
		"addl	$-2, %%eax;"             //eax: 3 * a - 2
		"division :;"
		"cltd;"
		"idivl %%r8d;"                  // eax: (a - b) / (a + b) or (3 * a - 2) / b
		"end:"
		// "retq;"
		:
	: [a] "r" (a), [b] "r" (b)
		: "eax", "r8d", "ecx", "rax"
		);
}

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
	asm volatile(
		"movl	%[a], %%eax;"            // eax: a
		"movl	%[b], %%r8d;"            // r8d: b
		"movl %%eax, %%ecx;"             // ecx: a  
		"subl %%r8d, %%ecx;"             // ecx: a - b
		"jle less_equal;"                // to a <= b
		"addl %%eax, %%r8d;"             // r8d: a + b
		"movl %%ecx, %%eax;"             // eax: a - b
		"jmp division;"                  // to end
		"less_equal :;"                  // if a <= b
		"jne less;"                      // to a < b
										 // if a == b
		"imull %%eax, %%eax;"            // eax: a^2
		"negl %%eax;"                    // eax: -(a^2)
		"jmp end;"                       // to exit
		"less :;"                        // if a < b
		"leal (%%rax, %%rax, 2), %%eax;" //eax: a + 2 * a
		"addl	$-2, %%eax;"             //eax: 3 * a - 2
		"division :;"
		"cltd;"
		"idivl %%r8d;"                  // eax: (a - b) / (a + b) or (3 * a - 2) / b
		"end:"
		"movl %%eax, %[result]"
		: [result] "=r" (result)
		: [a] "r" (a), [b] "r" (b)
		: "eax", "r8d", "ecx", "rax"
		);
	// f(a, b);
	// asm volatile("movl %%eax, %[result]\n\t" : [result] "=r" (result));
	std::cout << result << std::endl;
	// std::cout << result << std::endl;
	std::cout << f_cpp(a, b) << std::endl;
}