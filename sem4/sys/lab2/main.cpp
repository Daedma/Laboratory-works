#include <iostream>

int main()
{
	int result;
	int a = 1; // инициализируйте значения a, c, d заранее
	int c = 1;
	int d = 1;

	asm(
		"movl %[c], %%eax;" // помещаем с в eax для деления
		"sarl $2l, %%eax;" // делим с на 4
		"movl %[d], %%ebx;" // помещаем d в ebx для умножения
		"movl $28l, %%ecx;" // помещаем 28 в ecx
		"imul %%ecx, %%ebx;" // умножаем d на 28 и помещаем результат в ebx
		"addl %%ebx, %%eax;" // складываем c/4 и d*28, записываем в eax
		"cdq;"
		"push %%rax;" // помещаем в стек делимое
		"movl %[c], %%ecx;" // помещаем с в ecx
		"negl %%ecx;" // -с
		"decl %%ecx;" // -c - 1
		"movl %[a], %%eax;" // 
		"idivl %[d];" //
		"subl %%ecx, %%eax;" //
		"movl %%eax, %%ebx;"
		"pop %%rax;"
		"idiv %%ebx;"
		"movl %%eax, %[r];"
		: [r] "=r" (result)
		: [a] "r" (a), [c] "r" (c), [d] "r" (d)
		: "%eax", "%ecx", "%ebx", "%rax"
	);
	std::cout << result << std::endl;
	std::cout << (c / 4 + 28 * d) / (a / d - c - 1) << std::endl;
}

// asm(
// 	"movl %%eax, %1 \n\t"
// 	"cdql \n\t" // расширяет знак в edx:eax до 64 бит
// 	"idivl %4 \n\t" // выполняет целочисленное деление в edx:eax на 4, сохраняет остаток в edx
// 	"movl %%ebx, 28 \n\t"
// 	"imull %%edx, %%ebx \n\t" // умножает остаток от деления на 28 и сохраняет результат в edx
// 	"movl %%eax, %2 \n\t"
// 	"idivl %5 \n\t" // выполняет целочисленное деление в eax на d, сохраняет остаток в edx
// 	"subl %%eax, %3 \n\t"
// 	"subl %%eax, %1 \n\t" // вычитает c и a из eax
// 	"decl %%eax \n\t" // вычитает 1 из eax
// 	"cdql \n\t" // расширяет знак в edx:eax до 64 бит
// 	"idivl %%eax, %%edx \n\t" // делит edx:eax на eax и сохраняет результат в eax
// 	"movl %0, %%eax"
// 	: "=r"(result) // выходное значение записывается в переменную result
// 	: "r"(c), "r"(a), "r"(d), "r"(4), "r"(28)
// 	: "%eax", "%ebx", "%edx" // перечисляются используемые регистры
// );