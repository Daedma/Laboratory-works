#include <iostream>
#include <stdexcept>
#include <Windows.h>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>
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

		std::vector<HANDLE> m_pipes;

	public:
		ServerApplication(int argc, char const* argv[]);

		int run();

	private:

		void init_pipe();

		void run_childs();

		void send_lifetimes();

		void wait_childs();

		void console_input();

		void init_from_args();

	};
}

int main(int argc, char const* argv[])
{
	try
	{
		OS_Lab::ServerApplication app(argc, argv);
		return app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
}

void OS_Lab::ServerApplication::init_pipe()
{
	for (size_t i = 0; i != m_lifetimes.size(); ++i)
	{
		HANDLE pipe = CreateNamedPipeA(OSLAB_PIPENAME,
			PIPE_ACCESS_DUPLEX,
			PIPE_TYPE_MESSAGE | PIPE_WAIT | PIPE_READMODE_MESSAGE,
			PIPE_UNLIMITED_INSTANCES, sizeof(float), sizeof(float), 2000, NULL);
		if (pipe == INVALID_HANDLE_VALUE)
		{
			std::cerr << "Pipe creating failed\n";
		}
		m_pipes.emplace_back(pipe);
	}
}

void OS_Lab::ServerApplication::run_childs()
{
	m_pinfo.resize(m_lifetimes.size());
	for (auto& i : m_pinfo)
	{
		i.reset(new PROCESS_INFORMATION);
	}
	for (size_t i = 0; i != m_lifetimes.size(); ++i)
	{
		STARTUPINFOA cif;
		ZeroMemory(&cif, sizeof(STARTUPINFOA));
		cif.cb = sizeof(STARTUPINFOA);
		if (!CreateProcessA("./client.exe", NULL,
			NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &cif, m_pinfo[i].get()))
		{
			std::cerr << "Process creating has failure!\n";
		}
		else
		{
			std::cout << "Process #" << (i) << " has been created." << std::endl;
		}
	}
}

void OS_Lab::ServerApplication::send_lifetimes()
{
	for (size_t i = 0; i != m_lifetimes.size(); ++i)
	{
		message m{ m_lifetimes[i], i };
		ConnectNamedPipe(m_pipes[i], NULL);
		WriteFile(m_pipes[i], std::addressof(m), sizeof(message), NULL, NULL);
		FlushFileBuffers(m_pipes[i]);
		// std::clog << "Lifetime is sended to process number " << (i + 1) << " : " << m_lifetimes[i] << std::endl;
		DisconnectNamedPipe(m_pipes[i]);
	}
}

void OS_Lab::ServerApplication::wait_childs()
{
	for (HANDLE i : m_pipes)
	{
		DWORD openMode = PIPE_NOWAIT | PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE;
		SetNamedPipeHandleState(i, &openMode, NULL, NULL);
	}
	message m;
	while (!m_pipes.empty())
	{
		for (auto i = m_pipes.begin(); i != m_pipes.end(); ++i)
		{
			ConnectNamedPipe(*i, NULL);
			if (ReadFile(*i, &m, sizeof(message), NULL, NULL))
			{
				std::cout << "Process #" << m.number << " finished.\n";
				DisconnectNamedPipe(*i);
				CloseHandle(*i);
				i = m_pipes.erase(i);
				if (i == m_pipes.end()) break;
			}
		}
		// Sleep(100);
	}
	std::cout << "All processes were been finished.\n";
}

OS_Lab::ServerApplication::ServerApplication(int argc, char const* argv[])
{
	m_args.resize(argc);
	std::copy(argv, argv + argc, m_args.begin());
}

int OS_Lab::ServerApplication::run()
{
	if (m_args.size() > 2)
	{
		init_from_args();
	}
	else
	{
		console_input();
	}
	init_pipe();
	// std::cout << "pipe is inititialized\n";
	run_childs();
	// std::cout << "childs is runned\n";
	send_lifetimes();
	// std::cout << "lifetimes is sended\n";
	wait_childs();
	return EXIT_SUCCESS;
}

void OS_Lab::ServerApplication::init_from_args()
{
	size_t n = std::stoull(m_args[1]);
	if (m_args.size() - 2 < n)
	{
		throw std::invalid_argument{"Too less arguments"};
	}
	m_lifetimes.resize(n);
	std::transform(std::next(m_args.cbegin(), 2), m_args.cend(), m_lifetimes.begin(),
		[](const std::string& val) {
			float res = std::stof(val);
			if (res < 0)
			{
				throw std::invalid_argument{"Lifetime cannot be negative"};
			}
			if (std::abs(res) == INFINITY)
			{
				return DEFAULT_LIFETIME;
			}
			return res;
		});


}

void OS_Lab::ServerApplication::console_input()
{
	std::cout << "Enter number of processes: ";
	size_t n;
	std::cin.exceptions(std::ios::failbit);
	std::cin >> n;
	m_lifetimes.resize(n);
	for (size_t i = 0; i != n; ++i)
	{
		std::cout << "Enter the lifetime of process number " << i << " : ";
		std::string ts;
		std::cin >> ts;
		m_lifetimes[i] = std::stof(ts);
		if (m_lifetimes[i] < 0)
		{
			throw std::invalid_argument{"Lifetime cannot be negative"};
		}
		if (std::abs(m_lifetimes[i]) == INFINITY)
		{
			m_lifetimes[i] = DEFAULT_LIFETIME;
		}
	}
}