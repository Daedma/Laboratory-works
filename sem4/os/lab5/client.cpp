#include <Windows.h>
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

	std::set<Message> m_history;

	char* m_buffer = new char[BUFFER_SIZE];

};


int main(int argc, char const* argv[])
{
	/* code */
	return 0;
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
	std::cout << "Connected to server.";
	CreateThread(NULL, 0, receive_messages, this, NULL, NULL);
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
	std::memset(m_buffer, 0, BUFFER_SIZE);
	recv(m_sock, m_buffer, BUFFER_SIZE, NULL);
	Message message;
	message.from_bytes(m_buffer);
	std::cout << message << std::endl;
}

DWORD Client::receive_messages(LPVOID pclient)
{
	Client* client = (Client*)pclient;
	while (true)
	{
		int return_code = recv(client->m_sock, client->m_buffer, Client::BUFFER_SIZE, MSG_WAITALL);
		if (return_code == SOCKET_ERROR)
		{

		}
		Message new_message;
		new_message.from_bytes(client->m_buffer);
		WaitForSingleObject(client->m_history_mutex, INFINITE);
		client->m_history.emplace(new_message);
		ReleaseMutex(client->m_history_mutex);
	}
}

void Client::detach()
{

}