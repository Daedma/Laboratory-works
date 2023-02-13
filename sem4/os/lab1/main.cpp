#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char const* argv[])
{
	std::string str;
	if (argc > 1)
	{
		str = argv[1];
	}
	else
	{
		std::getline(std::cin, str);
	}
	std::replace(str.begin(), str.end(), ' ', ',');
	std::cout << str;
	return 0;
}