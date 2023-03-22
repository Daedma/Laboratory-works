#include "CalcParameters.hpp"
#include <iostream>
#include <cmath>
#include <signal.h>
#include <unistd.h>
#include <sstream>

class ServerApplication
{
	int stdin_desc;
	int stdout_desc;

public:
	ServerApplication();

	int run(int argc, const char* const* argv);

private:
	CalcParameters receive();

	CalcParameters calc(const CalcParameters& params);

	void send(const CalcParameters& result);

};

int main(int argc, char const* argv[])
{
	try
	{
		ServerApplication app;
		return app.run(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Server crashed with an error: " << e.what() << std::endl;
	}
}

int ServerApplication::run(int argc, const char* const* argv)
{
	std::clog << "Server start." << std::endl;
	std::clog << "Server pid: " << getpid() << std::endl;
	std::clog << "Server is waiting for data from the client..." << std::endl;
	CalcParameters params = receive();
	std::clog << "Server get data from client.\nX: " << params.x << "\nAccuracy: " << params.accuracy << std::endl;
	std::clog << "Server start calculation...\n";
	CalcParameters result = calc(params);
	std::clog << "Server has finished computing.\nSum: " << result.x << "\nAccuracy: " << result.accuracy << std::endl;
	std::clog << "Send results to client..." << std::endl;
	send(result);
	std::clog << "Data sent succesfully. Server exist." << std::endl;
	return EXIT_SUCCESS;
}

CalcParameters ServerApplication::receive()
{
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGUSR1);
	sigprocmask(SIG_BLOCK, &sigset, NULL);
	int sig;
	sigwait(&sigset, &sig);
	fdopen(STDIN_FILENO, "r");
	CalcParameters params;
	// std::cin >> params.x >> params.accuracy;
	std::cerr << "Server read from pipe: " << read(STDIN_FILENO, &params, sizeof(CalcParameters)) << std::endl;
	// fdopen(stdout_desc, "w");
	fclose(stdin);
	return params;
}

CalcParameters ServerApplication::calc(const CalcParameters& params)
{
	CalcParameters result{ 0., INFINITY };
	double last = std::sin(std::pow(params.x, 1)) / std::tgamma(3);
	for (size_t i = 0; result.accuracy > params.accuracy; ++i)
	{
		double cur = std::sin(std::pow(params.x, i)) / std::tgamma(i + 2);
		result.accuracy = std::abs(std::tgamma(i + 3) / std::tgamma(i + 2));
		if (result.accuracy > params.accuracy)
		{
			result.x += cur;
			last = cur;
		}
		std::cerr << "Iteration#" << i << std::endl;
	}
	return result;
}

void ServerApplication::send(const CalcParameters& result)
{
	fdopen(stdout_desc, "w");
	// std::cout << result.x << ' ' << result.accuracy;
	// std::cout.flush();
	std::cerr << "Server write: " << write(stdout_desc, &result, sizeof(CalcParameters)) << std::endl;
	// fflush(stdout);
	fsync(STDOUT_FILENO);
	fclose(stdout);
	signal(SIGUSR2, SIG_IGN);
	kill(getppid(), SIGUSR2);
}

ServerApplication::ServerApplication(): stdin_desc(fileno(stdin)), stdout_desc(fileno(stdout)) {}