#include <Windows.h>
#include <WinSock2.h>
#include <list>
#include <string>
#include <iostream>
#include "connection_info.hpp"
#include "message.hpp"

class Server
{
public:
	void init();

	void connect_clients();

	static Server* get_instance()
	{
		static Server instance;
		return &instance;
	}

private:
	SOCKET m_sock;

	std::list<SOCKET> m_clients;

	HANDLE m_out_mutex;

	HANDLE m_clients_mutex;

	struct ClientParams
	{
		SOCKADDR_IN addr;
		SOCKET sock;
	};

	static DWORD process_client(LPVOID addr);

	static std::string init_client(SOCKET sock);

	static void process_message(const Message& mess);
};

int main(int argc, char const* argv[])
{
	/* code */
	return 0;
}

void Server::init()
{
	m_out_mutex = CreateMutexA(NULL, FALSE, "stdout mutex");
	m_clients_mutex = CreateMutexA(NULL, FALSE, "clients mutex");
	SOCKADDR_IN addr;
	addr.sin_addr.s_addr = inet_addr(connection_info::ADDRESS);
	addr.sin_port = htons(connection_info::SERVER_PORT);
	addr.sin_family = AF_INET;
	m_sock = socket(AF_INET, SOCK_STREAM, NULL);
	bind(m_sock, (SOCKADDR*)&addr, sizeof(addr));
	listen(m_sock, connection_info::MAX_QUEUE_SIZE);
}

void Server::connect_clients()
{
	SOCKET accept_socket = SOCKET_ERROR;
	SOCKADDR_IN addr;
	int addr_size = sizeof(SOCKADDR_IN);
	while (true)
	{
		while (accept_socket == SOCKET_ERROR)
		{
			accept_socket = accept(m_sock, (SOCKADDR*)&addr, &addr_size);
		}
		WaitForSingleObject(m_clients_mutex, INFINITE);
		m_clients.emplace_back(accept_socket);
		ReleaseMutex(m_clients_mutex);
		ClientParams params{ addr, accept_socket };
		CreateThread(NULL, 0, process_client, (LPVOID)&params, 0, NULL);
	}
}

DWORD Server::process_client(LPVOID addr)
{
	Server* server = get_instance();
	ClientParams* params = (ClientParams*)addr;
	SOCKET sock = params->sock;
	std::string username = init_client(sock);
	char* buffer = new char[Message::MAX_SIZE];
	while (true)
	{
		int return_code = recv(sock, buffer, Message::MAX_SIZE, MSG_WAITALL);
		if (return_code == SOCKET_ERROR)
		{
			// Client detached
			WaitForSingleObject(server->m_clients_mutex, INFINITE);
			server->m_clients.erase(std::find(server->m_clients.begin(), server->m_clients.end(), sock));
			ReleaseMutex(server->m_clients_mutex);
			closesocket(sock);
			Message notify;
			notify.time = std::chrono::system_clock::now();
			notify.user = "Server";
			notify.content = username + " has been disconnected.";
			process_message(notify);
			delete[] buffer;
			return 0;
		}
		Message message;
		message.from_bytes(buffer);
		process_message(message);
	}
}

std::string Server::init_client(SOCKET sock)
{
	char* buffer = new char[Message::MAX_USERNAME_SIZE];
	int recv_bytes = recv(sock, buffer, Message::MAX_USERNAME_SIZE, MSG_WAITALL);
	Message notify;
	notify.time = std::chrono::system_clock::now();
	notify.user = "Server";
	notify.content = std::string{ buffer } + " has been connected.";
	process_message(notify);
	std::string name(buffer);
	delete[] buffer;
	return name;
}

void Server::process_message(const Message& mess)
{
	WaitForSingleObject(get_instance()->m_out_mutex, INFINITE);
	for (SOCKET i : Server::get_instance()->m_clients)
	{
		send(i, (char*)&mess, Message::MAX_SIZE, NULL);
	}
	std::cout << mess << std::endl;
	ReleaseMutex(get_instance()->m_out_mutex);
}