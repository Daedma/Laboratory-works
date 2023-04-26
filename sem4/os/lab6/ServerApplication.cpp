#include "Server.hpp"
#include <fstream>
#include <chrono>
#include <iostream>

std::string generateLogFileName()
{
	auto const time = std::chrono::current_zone()
		->to_local(std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
	return std::format("log_{:%Y_%m_%d_%Hh_%Mm_%Ss}.txt", time);
}

int main()
{
	Server chatServer;
	std::string filename = generateLogFileName();
	std::ofstream logFile(filename);
	chatServer.setLogOutput(logFile);
	chatServer.run();
}