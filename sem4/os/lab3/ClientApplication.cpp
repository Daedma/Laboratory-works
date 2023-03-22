#include <string>
#include <signal.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <atomic>
#include <iostream>
#include "CalcParameters.hpp"
#include <unistd.h>
#include <sstream>
#include <sys/wait.h>

class ClientApplication
{
	constexpr static const char* DEFAULT_INPUT_FILE = "input.txt";
	constexpr static const char* DEFAULT_OUTPUT_FILE = "output.txt";

	int stdin_desc;
	int stdout_desc;
	pid_t server_pid;

public:
	ClientApplication();

	int run(int argc, const char* const* argv);

private:
	CalcParameters input(const std::string& filename);

	void send(const CalcParameters& params);

	CalcParameters receive();

	void output(const std::string& filename, const CalcParameters& params);

	static pid_t to_pid(const char* bytes);
};

int main(int argc, char const* argv[])
{
	try
	{
		ClientApplication app;
		return app.run(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Client crashed with error: " << e.what() << std::endl;
	}
}


int ClientApplication::run(int argc, const char* const* argv)
{
	server_pid = to_pid(argv[0]);
	std::clog << "Client start." << std::endl;
	std::clog << "Client pid: " << getpid() << std::endl;
	std::string infile = argc > 1 ? argv[1] : DEFAULT_INPUT_FILE;
	std::clog << "Client started reading information from " << infile << "... " << std::endl;
	CalcParameters in_params = input(infile);
	std::clog << "Client read data from file.\nX: " << in_params.x << "\nAccuracy: " << in_params.accuracy << std::endl;
	std::clog << "Sending data to server..." << std::endl;
	send(in_params);
	std::clog << "Data sent to server." << std::endl;
	CalcParameters result = receive();
	std::clog << "Results received from server." << std::endl;
	std::clog << "Summ: " << result.x << "\nAccuracy: " << result.accuracy << std::endl;
	std::string outfile = argc > 2 ? argv[2] : DEFAULT_OUTPUT_FILE;
	output(outfile, result);
	std::clog << "Results recorded. Client exits." << std::endl;
	return EXIT_SUCCESS;
}



CalcParameters ClientApplication::input(const std::string& filename)
{
	CalcParameters params;
	std::ifstream ifs;
	ifs.exceptions();
	ifs.open(filename);
	if (ifs >> params.x >> params.accuracy)
		return params;
	return { NAN, NAN };
}

void ClientApplication::send(const CalcParameters& params)
{
	// FILE* fd = fdopen(STDOUT_FILENO, "w");
	// std::cerr << "fd == nullptr; " << (fd == nullptr) << '\n';
	// std::cout << params.x << " " << params.accuracy;
	// std::cout.flush();
	std::cerr << "Client write: " << write(STDOUT_FILENO, &params, sizeof(CalcParameters)) << " bytes\n";
	std::cerr << "EBADF: " << (errno & EBADF) << '\n'
		<< "EINVAL: " << (errno & EINVAL) << '\n'
		<< "EFAULT: " << (errno & EFAULT) << '\n'
		<< "EPIPE: " << (errno & EPIPE) << '\n'
		<< "EAGAIN: " << (errno & EAGAIN) << '\n'
		<< "EIO: " << (errno & EAGAIN) << '\n';
	fsync(STDOUT_FILENO);
	// fflush(stdout);
	fclose(stdout);
	// fclose(stdin);
	signal(SIGUSR1, SIG_IGN);
	// waitpid(-1, NULL, 0);
	kill(server_pid, SIGUSR1);
}

CalcParameters ClientApplication::receive()
{
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGUSR2);
	sigprocmask(SIG_BLOCK, &sigset, NULL);
	int tmp;
	sigwait(&sigset, &tmp);
	fdopen(STDIN_FILENO, "r");
	CalcParameters params;
	// std::cin >> params.x >> params.accuracy;
	std::cerr << "Client read: " << read(STDIN_FILENO, &params, sizeof(CalcParameters)) << std::endl;
	fclose(stdin);
	return params;
}

void ClientApplication::output(const std::string& filename, const CalcParameters& params)
{
	std::ofstream ofs;
	ofs.exceptions();
	ofs.open(filename);
	ofs.precision(17);
	ofs << "Summ: " << params.x << "\nAccuracy: " << params.accuracy;
}

pid_t ClientApplication::to_pid(const char* bytes)
{
	return std::stoi(bytes);
}

ClientApplication::ClientApplication(): stdin_desc(fileno(stdin)), stdout_desc(fileno(stdout)) {}