#include <iostream>
#include <thread>
#include <chrono>
#include "types.hpp"
#include <Windows.h>

int main(int argc, char const* argv[])
{
	std::cout << "Client started." << std::endl;
	HANDLE pipe = CreateFileA(OSLAB_PIPENAME, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
	if (pipe == INVALID_HANDLE_VALUE)
	{
		//TODO handle error
	}
	float lifetime;
	ReadFile(pipe, &lifetime, sizeof(lifetime), NULL, NULL);
	std::cout << "This procces will be exist for " << lifetime << " seconds." << std::endl;
	std::this_thread::sleep_for(std::chrono::duration<float>(lifetime));
	return 0;
}
