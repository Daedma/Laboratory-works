#include <iostream>

int main()
{
	int result;
	int a, c, d;
	std::cout << "Enter a: ";
	std::cin >> a;
	std::cout << "Enter c: ";
	std::cin >> c;
	std::cout << "Enter d: ";
	std::cin >> d;

	int overf = 0, zerrof = 0;

	__asm__ volatile (
		"xorl %[overf], %[overf]\n\t" // обнуляем флаг деления на ноль
		"xorl %[zerrof], %[zerrof]\n\t" // обнуляем флаг переполнение
		"movl %[c], %%eax\n\t" // помещаем c в регистр eax для деления
		"movl $4, %%ecx\n\t" // помещаем 4 в регистр ecx для деления
		"cltd\n\t" // == cdq; расширяем eax в edx:eax
		"idivl %%ecx\n\t" // производим деление с на 4, частное записывается в регистр eax, остаток в edx
		"imull $28, %[d], %%ecx\n\t" // умножаем d на 28, результат записывается в регистр ecx
		"jo error_of\n\t" // ошибка переполнения
		"addl %%ecx, %%eax\n\t" // eax: c / 4 + d * 28
		"jo error_of\n\t" // ошибка переполнения
		"movl %%eax, %%ebx\n\t" // ebx: с / 4 + d * 28
		"jo error_of\n\t" // ошибка переполнения
		"movl %[a], %%eax\n\t" // eax: a
		"cmpl $0, %[d]\n\t" // сравниваем d с нулем
		"jz error_zf\n\t" // ошибка деления на ноль
		"cltd\n\t" // расширяем eax до edx:eax
		"idivl %[d]\n\t" // eax: a / d, edx: a % d
		"movl %%eax, %%ecx\n\t" // ecx: a / d
		"movl %%ebx, %%eax\n\t" // eax: c / 4 + d * 28
		"subl %[c], %%ecx\n\t" // ecx: a / d - c
		"jo error_of\n\t" // ошибка переполнения
		"subl $1, %%ecx\n\t" // ecx: a / d - c - 1
		"jz error_zf\n\t" // ошибка деления на ноль
		"jo error_of\n\t" // ошибка переполнения
		"cltd\n\t" // расширяем eax до edx:eax
		"idivl %%ecx\n\t" // eax: (c / 4 + 28 * d) / (a / d - c - 1), edx: (c / 4 + 28 * d) % (a / d - c - 1)
		"movl %%eax, %[r]\n\t" // result: (c / 4 + 28 * d) / (a / d - c - 1)
		"jmp exit\n\t" // выход из вставки
		"error_of:\n\t" // обработка ошибки переполнения
		"movl $1, %[overf]\n\t" // overf: 1
		"jmp exit\n\t" // выход из вставки
		"error_zf:\n\t" // обработка ошибки деления на ноль
		"movl $1, %[zerrof]\n\t" // zerrof: 1
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
		std::cout << "Asm result: " << result << std::endl;
		std::cout << "Cpp result: " << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
	}
}