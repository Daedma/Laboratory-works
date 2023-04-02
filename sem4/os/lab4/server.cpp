#include <iostream>
#include <stdexcept>
#include <Windows.h>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "types.hpp"

// С процесса - сервера запускается n процессов клиентов.
// Для каждого из созданных клиентов указывается время жизни(в секундах).
// Клиент запускается, существует заданное время и завершает работу.
// Также следует предусмотреть значение для бесконечного времени.
// Требуется не менее трех одновременно запускаемых процессов - клиентов.

namespace OS_Lab
{

	class ServerApplication
	{
		std::vector<float> m_lifetimes;

		std::vector<std::string> m_args;

		std::vector<std::unique_ptr<PROCESS_INFORMATION>> m_pinfo;

		HANDLE m_pipe;

	public:
		ServerApplication(int argc, char const* argv[]);

		int run();

	private:
		void get_childs_number();

		void get_childs_lifetimes();

		void init_pipe();

		void run_childs();

		void send_lifetimes();

		void wait_childs();

		std::wstring get_pipename(size_t n)
		{
			return LR"(\\OSLABServer\pipe\)" + std::to_wstring(n);
		}
	};
}

int main(int argc, char const* argv[])
{
	OS_Lab::ServerApplication app(argc, argv);
	return app.run();
}

void OS_Lab::ServerApplication::init_pipe()
{
	HANDLE pipe = CreateNamedPipe(get_pipename(i).c_str(),
		PIPE_ACCESS_DUPLEX,
		PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE,
		PIPE_WAIT, sizeof(message), sizeof(message),
		0, NULL);
	// for (size_t i = 0; i != m_lifetimes.size(); ++i)
	// {
	// 	HANDLE pipe = CreateNamedPipe(get_pipename(i).c_str(),
	// 		PIPE_ACCESS_DUPLEX,
	// 		PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE,
	// 		PIPE_WAIT, sizeof(message), sizeof(message),
	// 		0, NULL);
	// 	m_pipes.emplace_back(pipe);
	// }
}

void OS_Lab::ServerApplication::run_childs()
{
	for (size_t i = 0; i != m_lifetimes.size(); ++i)
	{
		STARTUPINFO cif;
		ZeroMemory(&cif, sizeof(STARTUPINFO));
		cif.cb = sizeof(STARTUPINFO);
		if (!CreateProcess(L"client.exe", get_pipename(i).data(),
			NULL, NULL, FALSE, NULL, NULL, NULL, &cif, m_pinfo[i].get()))
		{
			//TODO handle error
		}
	}
}

void OS_Lab::ServerApplication::send_lifetimes()
{
	for (size_t i = 0; i != m_lifetimes.size(); ++i)
	{
		ConnectNamedPipe()
			WriteFile(m_pipes[i], std::addressof(m_lifetimes[i]), sizeof(float), NULL, NULL);
	}
}

void OS_Lab::ServerApplication::wait_childs()
{

}