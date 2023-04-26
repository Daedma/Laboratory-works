#include "Client.hpp"
#include <fstream>
#include <thread>
#include <functional>
#include <iostream>

Client::Client(const std::string& config)
{
	initFromConfig(config);
	m_sock.setBlocking(true);
}

void Client::initFromConfig(const std::string& config)
{
	std::ifstream ifs;
	ifs.exceptions(std::ios::failbit);
	ifs.open(config);

	std::string ipString;
	uint16_t port;
	ifs >> ipString >> port;

	ifs.close();

	m_serverIp = ipString;
	m_serverPort = port;
}

void Client::joinAs(const std::string& username)
{
	sf::Socket::Status status = m_sock.connect(m_serverIp, m_serverPort);
	if (status != sf::Socket::Status::Done)
	{
		// TODO handle error
		std::cerr << "Failed to connect to server\n";
	}
	else
	{
		// TODO print about connection
		sf::Packet pack;
		m_sock.receive(pack);
		// TODO print joined to chat
		m_inChat = true;
		sf::Int8 qCode;
		pack >> qCode;
		pack.clear();
		pack << username;
		m_sock.send(pack);
		m_username = username;
		std::thread{std::mem_fn(&Client::receiveMessages), this}.detach();
	}
}

void Client::receiveMessages()
{
	while (true)
	{
		sf::Packet pack;
		if (m_sock.receive(pack) != sf::Socket::Status::Done)
		{
			// TODO handle error
			m_inChat = false;
		}
		else
		{
			Message message;
			pack >> message;
			m_historyMutex.lock();
			m_history.emplace(std::move(message));
			m_historyMutex.unlock();
			m_isDirty = true;
		}
	}
}

void Client::sendMessage(const std::string& content)
{
	Message message;
	message.time = std::chrono::system_clock::now();
	message.user = m_username;
	message.content = content;
	sf::Packet pack;
	pack << message;
	m_sock.send(pack);
}

void Client::detach()
{
	m_inChat = false;
	m_sock.disconnect();
}