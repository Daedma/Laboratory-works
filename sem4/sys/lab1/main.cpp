#include <iostream>

int main(int argc, char const* argv[])
{
	int a, b;
	std::cin >> a >> b;
	int sum;
	__asm__(
		"xorl %%eax, %%eax;"
		"movl %[a], %%eax;"
		"addl %[b], %%eax;"
		"movl %%eax, %[s];"
		: [s] "=r"(sum)
		: [a] "r"(a), [b]"r"(b)
		: "%eax", "%ebx"
	);
	std::cout << "Sum: " << sum << '\n';
	return 0;
}
