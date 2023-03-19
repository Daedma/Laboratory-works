#include "CalcParameters.hpp"
#include <iostream>
#include <cmath>

class ServerApplication
{
public:
	int run(int argc, const char* const* argv);

private:
	CalcParameters receive();

	CalcParameters calc(const CalcParameters& params);

	void send(const CalcParameters& result);

};

int main(int argc, char const* argv[])
{
	ServerApplication app;
	return app.run(argc, argv);
}

int ServerApplication::run(int argc, const char* const* argv)
{
	CalcParameters accuracy = receive();
	CalcParameters result = calc(accuracy);
	send(result);
	return EXIT_SUCCESS;
}

CalcParameters ServerApplication::receive()
{
	CalcParameters params;
	std::cin >> params.x >> params.accuracy;
	return params;
}

CalcParameters ServerApplication::calc(const CalcParameters& result)
{

	return CalcParameters();
}

void ServerApplication::send(const CalcParameters& result)
{
	std::cout << result.x << ' ' << result.accuracy;
}

