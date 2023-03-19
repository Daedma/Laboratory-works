#include <string>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

class ClientApplication
{
	constexpr static char* DEFAULT_INPUT_FILE = "input.txt";
	constexpr static char* DEFAULT_OUTPUT_FILE = "output.txt";

public:
	ClientApplication();

	int run(int argc, const char* const* argv);

	struct CalcParameters
	{
		double x;
		double accuracy;
	};

private:
	double input(const std::string& filename);

	void send(double accuracy);

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
	double in_accuracy = input(infile);
	send(in_accuracy);
	CalcParameters result = receive();
	std::string outfile = argc > 2 ? argv[2] : DEFAULT_OUTPUT_FILE;
	output(outfile, result);
}



double ClientApplication::input(const std::string& filename)
{
	double accuracy;
	std::ifstream ifs;
	ifs.exceptions();
	ifs.open(filename);
	if (ifs >> accuracy)
		return accuracy;
	return NAN;
}

void ClientApplication::send(double accuracy)
{
	std::cout << accuracy;
}

ClientApplication::CalcParameters ClientApplication::receive()
{
	CalcParameters params;
	std::cin >> params.x >> params.accuracy;
	return params;
}

void ClientApplication::output(const std::string& filename, const ClientApplication::CalcParameters& params)
{
	std::ofstream ofs;
	ofs.exceptions();
	ofs.open(filename);
	ofs << "X: " << params.x << "\nAccuracy: " << params.accuracy;
}