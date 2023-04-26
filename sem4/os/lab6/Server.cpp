#include "Server.hpp"
#include "Queries.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <functional>
#include <format> // c++20 is required

Server::Server(const std::string& config) :
	m_semaphore(3)
{
	initFromConfig(config);
	m_listener.setBlocking(true);
}

void Server::initFromConfig(const std::string& config)
{
	std::ifstream ifs;
	ifs.exceptions(std::ios::failbit);
	ifs.open(config);

	std::string ipString;
	uint16_t port;
	ifs >> ipString >> port;

	ifs.close();

	m_ip = ipString;
	m_port = port;
}

void Server::run()
{
	m_listener.listen(m_port);
	acceptClients();
}

void Server::acceptClients()
{
	while (true)
	{
		sf::TcpSocket* newClient = new sf::TcpSocket();
		sf::Socket::Status status = m_listener.accept(*newClient);
		if (status != sf::Socket::Done)
		{
			makeLog("An error occurred while connecting a new client.");
		}
		else
		{
			handleNewConnection(newClient);
		}
	}

}

void Server::handleNewConnection(sf::TcpSocket* client)
{
	makeLog(std::format("Client [{}:{}] is connected to the server.",
		client->getRemoteAddress().toString(), client->getRemotePort()));
	m_clientsMutex.lock();
	m_clients.emplace_back(client);
	m_clientsMutex.unlock();
	++m_activeConnections;
	std::thread{std::mem_fn(&Server::processClient), this, client}.detach();
}

void Server::processClient(sf::TcpSocket* client)
{
	m_semaphore.acquire();
	std::string username = addToChat(client);
	client->setBlocking(true);
	Message message;
	while (true)
	{
		sf::Packet pack;
		sf::Socket::Status status = client->receive(pack);
		if (status != sf::Socket::Done)
		{
			// Client disconnected
			makeLog("Lost connection to user " + username);
			removeFromChat(client, username);
			detachClient(client);
			m_semaphore.release();
			makeLog(std::format("User {} successfully removed from chat", username));
			makeLog(std::format("Current number of users in the chat: {}\nQueue size: {}",
				static_cast<size_t>(m_usersInChat),
				static_cast<size_t>(m_activeConnections - m_usersInChat)));
			return;
		}
		pack >> message;
		processMessage(message);
	}
}

std::string Server::addToChat(sf::TcpSocket* client)
{
	makeLog(std::format("Add client [{}:{}] to chat.",
		client->getRemoteAddress().toString(), client->getRemotePort()));
	sf::Packet pack;
	pack << Query::Code::USERNAME;
	client->send(pack);
	pack.clear();
	client->receive(pack);
	std::string username;
	pack >> username;
	m_inChatMutex.lock();
	m_inChat.emplace_back(client);
	m_inChatMutex.unlock();
	++m_usersInChat;
	makeLog(std::format("{} has joined the chat", username));
	Message message;
	message.time = std::chrono::system_clock::now();
	message.user = "Server";
	message.content = username + " has been connected.";
	processMessage(message);
	return username;
}

void Server::removeFromChat(sf::TcpSocket* client, const std::string& username)
{
	m_inChatMutex.lock();
	m_inChat.remove(client);
	m_inChatMutex.unlock();
	--m_usersInChat;
	Message message;
	message.time = std::chrono::system_clock::now();
	message.user = "Server";
	message.content = username + " has been disconnected.";
	processMessage(message);
}

void Server::detachClient(sf::TcpSocket* client)
{
	m_clientsMutex.lock();
	m_clients.remove_if([client](const auto& val) {return val.get() == client;});
	m_clientsMutex.unlock();
	--m_activeConnections;
}

void Server::processMessage(const Message& message)
{
	sf::Packet pack;
	pack << message;
	m_inChatMutex.lock();
	for (auto& i : m_inChat)
	{
		i->send(pack);
	}
	m_inChatMutex.unlock();
	makeLog("(New message) " + message.toString());
}

void Server::makeLog(const std::string& message)
{
	m_logOutMutex.lock();
	if (m_duplicateToStderr)
	{
		std::clog << message << std::endl;
	}
	if (m_logOut)
	{
		(*m_logOut) << message << std::endl;
	}
	m_logOutMutex.unlock();
}