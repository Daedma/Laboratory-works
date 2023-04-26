#include "Client.hpp"
#include <iostream>

const char* const command_list =
"List of available commands:\n"
"\t\\h - print this list\n"
"\t\\u - update history\n"
"\t\\w - write new message\n"
"\t\\d - disconnect\n";

int main(int argc, char const* argv[])
{
	try
	{
		std::string username;
		std::cout << "Please, enter your username\n>";
		std::cin >> username;
		Client client;
		client.joinAs(username);
		std::cout << "You have successfully joined the chat!" << std::endl;
		std::string command;
		std::string message;
		while (client.isInChat())
		{
			std::cout << "Please, enter command\n>";
			std::cin >> command;
			if (command == "\\u")
			{
				client.processChat([](const Message& message) {
					std::cout << message.toString() << std::endl;
					});
			}
			else if (command == "\\h")
			{
				std::cout << command_list;
			}
			else if (command == "\\w")
			{
				std::getline(std::cin, message);
				client.sendMessage(message);
			}
			else if (command == "\\d")
			{
				client.detach();
				return 0;
			}
			else
			{
				std::cout << "Undefined command.\n";
				std::cout << command_list;
			}
		}
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		system("pause");
		return EXIT_FAILURE;
	}
}
