// #include <Windows.h>
#include <iostream>
#include "connection_info.hpp"
#include "message.hpp"
#include <WinSock2.h>
#include <string>
#include <stdexcept>
#include <chrono>
#include <set>

// Вариант 23
// Разработать чат для обмена сообщениями. Пусть на сервере есть чат, к которому могут 
// одновременно присоединяться только 3 процесса - клиента.
// Остальные ждут своей очереди.Чат общий для всех, то есть при подключении, 
// отключении клиента и появлении нового 
// сообщения информация об это рассылается по всем подключенным клиентам, но 
// старая история для вновь подключившегося клиента не отсылается.



class Client
{
public:
	Client()
	{};

	~Client()
	{
		delete[] m_buffer;
		// shutdown(m_sock, SD_SEND);
		closesocket(m_sock);
	}

	void join(const std::string& username);

	void sendMessage(const std::string& message);

	void detach();

	void update();

	class connection_error : public std::runtime_error
	{
	public:
		connection_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
		connection_error(const char* what_arg) : std::runtime_error(what_arg) {}
	};

	inline static constexpr size_t BUFFER_SIZE = Message::MAX_SIZE;

private:
	static DWORD receive_messages(LPVOID pclient);

	SOCKET m_sock;


	std::string m_username;

	HANDLE m_history_mutex;

	std::set<Message, Message::less> m_history;

	char* m_buffer = new char[BUFFER_SIZE];

	bool get_status();

};

const char* const command_list =
"List of available commands:\n"
"\t\\h - print this list\n"
"\t\\u - update history\n"
"\t\\w - write new message\n"
"\t\\d - disconnect\n";


int main(int argc, char const* argv[])
{
	WSADATA wsaData;
	int iResult;
	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0)
	{
		printf("WSAStartup failed: %d\n", iResult);
		return 1;
	}
	try
	{
		std::string username;
		std::cout << "Please, enter your username\n>";
		do
		{
			std::cin >> username;
		} while (username.size() > Message::MAX_USERNAME_SIZE);
		Client client;
		client.join(username);
		std::cout << "You have successfully joined the chat!" << std::endl;
		std::string command;
		std::string message;
		while (true)
		{
			std::cout << "Please, enter command\n>";
			std::cin >> command;
			if (command == "\\u")
			{
				client.update();
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
				Sleep(1000);
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
	WSACleanup();
}

void Client::join(const std::string& username)
{
	SOCKADDR_IN addr;
	addr.sin_addr.s_addr = inet_addr(connection_info::ADDRESS);
	addr.sin_port = htons(connection_info::SERVER_PORT);
	addr.sin_family = AF_INET;
	m_sock = socket(AF_INET, SOCK_STREAM, NULL);
	std::cout << "Connecting to server...\n";
	if (connect(m_sock, (SOCKADDR*)&addr, sizeof(SOCKADDR_IN)) != 0)
	{
		m_sock = NULL;
		throw connection_error{"failed connect to server."};
	}
	m_username = username;
	m_username.resize(Message::MAX_USERNAME_SIZE);
	send(m_sock, m_username.data(), Message::MAX_USERNAME_SIZE, NULL);
	std::cout << "Connected to server.\n";
	if (get_status())
	{
		CreateThread(NULL, 0, receive_messages, this, NULL, NULL);
	}
	else
	{
		m_sock = NULL;
		throw connection_error{"failed connect to server."};
	}
}

void Client::sendMessage(const std::string& message)
{
	Message mess;
	mess.time = std::chrono::system_clock::now();
	mess.user = m_username;
	mess.content = message;
	char* buff = new char[BUFFER_SIZE];
	mess.to_bytes(buff);
	send(m_sock, buff, BUFFER_SIZE, NULL);
}

void Client::update()
{
	WaitForSingleObject(m_history_mutex, INFINITE);
	for (const auto& i : m_history)
	{
		std::cout << i << std::endl;
	}
	ReleaseMutex(m_history_mutex);
}

DWORD Client::receive_messages(LPVOID pclient)
{
	Client* client = (Client*)pclient;
	while (true)
	{
		std::memset(client->m_buffer, 0, BUFFER_SIZE);
		int return_code = recv(client->m_sock, client->m_buffer, Client::BUFFER_SIZE, MSG_WAITALL);
		if (return_code == SOCKET_ERROR)
		{
			std::cout << "Connection to server lost." << std::endl;
			return 0;
		}
		Message new_message;
		new_message.from_bytes(client->m_buffer);
		WaitForSingleObject(client->m_history_mutex, INFINITE);
		client->m_history.emplace(std::move(new_message));
		ReleaseMutex(client->m_history_mutex);
	}
}

void Client::detach()
{
	// shutdown(m_sock, SD_SEND);
	closesocket(m_sock);
}

bool Client::get_status()
{
	int status = 0;
	recv(m_sock, (char*)&status, sizeof(int), NULL);
	return status == 1;
}