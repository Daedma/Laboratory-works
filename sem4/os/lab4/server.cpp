#include <iostream>
#include <stdexcept>
#include <Windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include "types.hpp"

// С процесса - сервера запускается n процессов клиентов.
// Для каждого из созданных клиентов указывается время жизни(в секундах).
// Клиент запускается, существует заданное время и завершает работу.
// Также следует предусмотреть значение для бесконечного времени.
// Требуется не менее трех одновременно запускаемых процессов - клиентов.

void run_child(float lifetime, size_t number)
{
	OpenSemaphore(EVENT_ALL_ACCESS, FALSE, SEMAPHORE_NAME);

}

void run_childs(size_t n, size_t max, const std::vector<float>& lifetimes)
{
	std::vector<float> sorted_lt = lifetimes;
	std::sort(sorted_lt.begin(), sorted_lt.end());

	CreateSemaphore(NULL, 0, max, SEMAPHORE_NAME);
	for (size_t i = 0; i != n; ++i)
	{
		run_child(sorted_lt[i], i + 1);
	}
}

int main(int argc, char const* argv[])
{
	if (argc > 2)
	{
		size_t n = std::stoull(argv[1]); //TODO handle exception
		if (argc < n + 3)
		{
			throw std::invalid_argument{"too less arguments"};
		}
		size_t max = std::stoull(argv[2]);
		std::vector<float> lifetimes;
		for (size_t i = 3; i != n + 3; ++i)
		{
			lifetimes.emplace_back(std::stof(argv[i]));
		}
		std::vector<HANDLE> pipes;
		for (size_t i = 0; i != max; ++i)
		{
			std::wstring pipe_name = LR"(\\LAB4Server\pipe\)" + std::to_wstring(i);
			HANDLE pipe = CreateNamedPipe(pipe_name.c_str(),
				PIPE_ACCESS_OUTBOUND,
				PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE,
				PIPE_NOWAIT, sizeof(message), sizeof(message),
				0, NULL);
			pipes.emplace_back(pipe);
		}
		run_childs(n, max, lifetimes);
	}
	else
	{
		// TODO print usage info
	}
	return 0;
}
