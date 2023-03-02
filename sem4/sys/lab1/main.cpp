#include <iostream>

int main(int argc, char const* argv[])
{
	int a, b;
	std::cin >> a >> b;
	int sum = 134;
	__asm__(
		// "xor %%eax, %%eax;"
		// "mov %%ebx, %%eax;"
		// "mov %%ebx, %1;"
		// "add %%ebx, %2;"
		// "mov %0, %%ebx;"
		"mov %%eax, $12;"
		"mov %0, %%eax"
		: "=r" (sum)
		: "r" (a), "r" (b)
	);
	std::cout << "Sum: " << sum << '\n';
	return 0;
}
