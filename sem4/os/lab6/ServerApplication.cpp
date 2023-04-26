#include "Server.hpp"
#include <fstream>

int main()
{
	Server chatServer;
	std::ofstream logFile("server_log.txt");
	chatServer.setLogOutput(logFile);
	chatServer.run();
}