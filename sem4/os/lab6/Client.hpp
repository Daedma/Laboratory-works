#pragma once

#include "Queries.hpp"
#include "Message.hpp"
#include <SFML/Network.hpp>
#include <set>
#include <string>
#include <mutex>
#include <atomic>
#include <algorithm>

class Client
{
	sf::TcpSocket m_sock;

	sf::Uint16 m_serverPort;

	sf::IpAddress m_serverIp;

	std::string m_username;

	mutable std::mutex m_historyMutex;

	std::set<Message, Message::less> m_history;

	std::atomic_bool m_inChat = false;

	std::atomic_bool m_isDirty = false;

public:
	Client(const std::string& config = "server.ini");

	bool joinAs(const std::string& username);

	void setUsername(const std::string& username) { m_username = username; }

	bool isAvailableName() const { return m_username.size() >= 3 && m_username.size() <= 8; }

	bool join() { return joinAs(m_username); }

	void sendMessage(const std::string& content);

	bool isDirty() const { return m_isDirty; }

	bool isInChat() const { return m_inChat; }

	template<typename Func>
	void processChat(Func f)
	{
		m_historyMutex.lock();
		std::for_each(m_history.cbegin(), m_history.cend(), f);
		m_historyMutex.unlock();
		m_isDirty = false;
	}

	void detach();

private:
	void initFromConfig(const std::string& config);

	void receiveMessages();
};