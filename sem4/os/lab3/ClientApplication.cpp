#include <string>
#include <signal.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <atomic>
#include <iostream>
#include "CalcParameters.hpp"

class ClientApplication
{
	constexpr static const char* DEFAULT_INPUT_FILE = "input.txt";
	constexpr static const char* DEFAULT_OUTPUT_FILE = "output.txt";

	int stdin_desc;
	int stdout_desc;

public:
	ClientApplication();

	int run(int argc, const char* const* argv);

private:
	CalcParameters input(const std::string& filename);

	void send(const CalcParameters& params);

	CalcParameters receive();

	void output(const std::string& filename, const CalcParameters& params);
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
	std::clog << "Client start." << std::endl;
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
	fdopen(stdout_desc, "w");
	std::cout << params.x << ' ' << params.accuracy;
	fclose(stdout);
	signal(SIGUSR1, SIG_IGN);
	killpg(0, SIGUSR1);
	// fclose(stdin);
}

CalcParameters ClientApplication::receive()
{
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGUSR2);
	int tmp;
	sigwait(&sigset, &tmp);
	fdopen(stdin_desc, "r");
	CalcParameters params;
	std::cin >> params.x >> params.accuracy;
	fclose(stdin);
	return params;
}

void ClientApplication::output(const std::string& filename, const CalcParameters& params)
{
	std::ofstream ofs;
	ofs.exceptions();
	ofs.open(filename);
	ofs << "Summ: " << params.x << "\nAccuracy: " << params.accuracy;
}

ClientApplication::ClientApplication(): stdin_desc(fileno(stdin)), stdout_desc(fileno(stdout)) {}