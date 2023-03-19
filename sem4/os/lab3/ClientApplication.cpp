#include <string>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include "CalcParameters.hpp"

class ClientApplication
{
	constexpr static char* DEFAULT_INPUT_FILE = "input.txt";
	constexpr static char* DEFAULT_OUTPUT_FILE = "output.txt";

public:
	int run(int argc, const char* const* argv);


private:
	CalcParameters input(const std::string& filename);

	void send(const CalcParameters& params);

	CalcParameters receive();

	void output(const std::string& filename, const CalcParameters& params);
};

int main(int argc, char const* argv[])
{
	ClientApplication app;
	return app.run(argc, argv);
}


int ClientApplication::run(int argc, const char* const* argv)
{
	std::string infile = argc > 1 ? argv[1] : DEFAULT_INPUT_FILE;
	CalcParameters in_params = input(infile);
	send(in_params);
	CalcParameters result = receive();
	std::string outfile = argc > 2 ? argv[2] : DEFAULT_OUTPUT_FILE;
	output(outfile, result);
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
	std::cout << params.x << ' ' << params.accuracy;
}

CalcParameters ClientApplication::receive()
{
	CalcParameters params;
	std::cin >> params.x >> params.accuracy;
	return params;
}

void output(const std::string& filename, const CalcParameters& params)
{
	std::ofstream ofs;
	ofs.exceptions();
	ofs.open(filename);
	ofs << "Summ: " << params.x << "\nAccuracy: " << params.accuracy;
}