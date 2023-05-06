#include <iostream>

int f_cpp(int* a, int sz, int d)
{
	int res = 0;
	for (size_t i = 0; i != sz; ++i)
	{
		if (a[i] < 0 && a[i] <= d)
		{
			res += a[i] * a[i] * a[i];
		}
	}
	return res;
}

int main()
{
	int n;
	std::cout << "Enter size of massive\n>";
	std::cin >> n;
	int* a = new int[n];
	for (int i = 0; i != n; ++i)
	{
		std::cout << "[" << i << "] : ";
		std::cin >> a[i];
	}
	int d;
	std::cout << "Enter d\n>";
	std::cin >> d;
	int res = 0;
	int error = 0;
	asm volatile(
		"xor %%rcx, %%rcx;"
		"xor %%rsi, %%rsi;"            // rsi = 0, индексная переменная
		"mov %[size], %%ecx;"          // ecx = size, размер массива
		"jecxz exit;"                  // если массив пуст, то выход из цикла
		"begin_loop:"                  // начало цикла
		"movl (%[a], %%rsi, 0x4), %%eax;" // eax = a[i]
		"test %%eax, %%eax;"
		"jns end_loop;"                // если текущий элемент неотрицательный, то переход к следующему
		"cmp %[d], %%eax;"
		"jg end_loop;"                 // если текущий элемент больше d, то переход к следующему
		"mov %%eax, %%ebx;"
		"imul %%eax, %%ebx;"           // возведение в куб текущего элемента
		"jo error;"
		"imul %%ebx, %%eax;"
		"jo error;"
		"add %%eax, %[res];"           // акумулируем результат в res
		"jo error;"
		"end_loop:"
		"inc %%rsi;"                   // инкрементируем счетчик
		"loop begin_loop;"
		"jmp exit;"                    // выход из цикла
		"error:"                       // обработка ошибки переполнения
		"movl $1, %[err];"
		"exit:;"
		: [res] "+m" (res), [err] "=m" (error)
		: [a] "r" (*(const int(*)[]) a), [d] "r" (d), [size] "m" (n)
		: "rax", "rsi", "rcx", "eax", "memory", "ebx"
		);
	for (int i = 0; i != n; ++i)
	{
		std::cout << a[i] << ' ';
	}
	std::cout << std::endl;
	if (error)
	{
		std::cout << "Overflow!\n";
	}
	else
	{
		std::cout << "asm : " << res << std::endl;
	}
	std::cout << "cpp : " << f_cpp(a, n, d) << std::endl;
}