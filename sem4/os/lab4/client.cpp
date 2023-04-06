#include <iostream>
#include <thread>
#include <chrono>
#include "types.hpp"
#include <Windows.h>

int main(int argc, char const* argv[])
{
	std::cout << "Client started." << std::endl;
	WaitNamedPipeA(OSLAB_PIPENAME, 10000);
	HANDLE pipe = CreateFileA(OSLAB_PIPENAME, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
	if (pipe == INVALID_HANDLE_VALUE)
	{
		std::cerr << "Failure to open pipe." << std::endl;
		return -1;
	}
	message m;
	ReadFile(pipe, &m, sizeof(message), NULL, NULL);
	CloseHandle(pipe);
	std::cout << "Process #" << m.number << " will be exist for " << m.lifetime << " seconds." << std::endl;
	std::this_thread::sleep_for(std::chrono::duration<float>(m.lifetime));
	pipe = CreateFileA(OSLAB_PIPENAME, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
	WriteFile(pipe, &m, sizeof(message), NULL, NULL);
	std::cout << "This process finished.\n";
	CloseHandle(pipe);
	return 0;
}
