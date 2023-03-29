#include <iostream>

int f(int a, int b)
{
	int error;
	asm volatile(
		"movl	%[a], %%ecx\n\t" // ecx: a
		"movl	%[b], %%esi\n\t" // esi: b
		"cmpl %%esi, %%ecx\n\t"
		"jle .L2\n\t"
		//if a > b:
		"movl %%ecx, %%eax\n\t" // eax: a
		"addl %%esi, %%ecx\n\t" // ecx: a + b
		"subl %%esi, %%eax\n\t" // eax: a - b
		"cltd\n\t" // cdq
		"idivl %%ecx\n\t" // (a - b) / (a + b)
		// "movl %%eax, %[result]\n\t"
		".L3:\n\t"
		// "xorl % eax, % eax\n\t"
		"ret\n\t" // выход из функции
		".L2:\n\t"
		// if a <= b:
		"je .L6\n\t"
		"jge .L3\n\t"
		// "leal(% rcx, % rcx, 2), % eax\n\t"
		"subl	$2, %%eax\n\t"
		"cltd\n\t"
		"idivl %%esi\n\t"
		// "movl %%eax, result(% rip)\n\t"
		"jmp .L3\n\t"
		".L6:\n\t"
		// if a == b:
		"movl %%ecx, %%eax\n\t"
		"negl %%eax\n\t"
		"imull %%eax, %%ecx\n\t"
		// "movl % ecx, result(% rip)\n\t"
		"jmp .L3\n\t"
		: [error] "=r"(error)
		: [a] "r" (a), [b] "r" (b)
		: "eax", "ebx", "edx", "esi"
		);
}