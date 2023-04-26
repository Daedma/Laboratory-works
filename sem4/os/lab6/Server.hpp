#pragma once
#include <list>
#include <string>
#include <semaphore> // C++20 required
#include <mutex>
#include <atomic>
#include <utility>
#include <memory>
#include <iosfwd>
#include <SFML\Network.hpp>
#include "Message.hpp"

class Server
{
	// Constants

	static constexpr size_t MAX_CHAT_USERS = 3ULL;

	// Network

	sf::TcpListener m_listener;

	sf::IpAddress m_ip;

	sf::Uint16 m_port;

	std::list<std::unique_ptr<sf::TcpSocket>> m_clients;

	std::list<sf::TcpSocket*> m_inChat;

	std::atomic_size_t m_usersInChat = 0;

	std::atomic_size_t m_activeConnections = 0;

	// Synchronization

	std::counting_semaphore<MAX_CHAT_USERS> m_semaphore;

	std::mutex m_inChatMutex;

	std::mutex m_clientsMutex;

	std::mutex m_logOutMutex;

	// Loging

	std::ostream* m_logOut = nullptr;

	bool m_duplicateToStderr = true;
public:
	Server(const std::string& config = "server.ini");

	void setLogOutput(std::ostream& os) { m_logOut = &os; }

	void setLogToStderr(bool shouldLogToStderr) { m_duplicateToStderr = shouldLogToStderr; }

	void run();

private:
	void initFromConfig(const std::string& config);

	void handleNewConnection(sf::TcpSocket* client);

	void processClient(sf::TcpSocket* client);

	void acceptClients();

	void makeLog(const std::string& message);

	std::string addToChat(sf::TcpSocket* client);

	void removeFromChat(sf::TcpSocket* client, const std::string& username);

	void detachClient(sf::TcpSocket* client);

	void processMessage(const Message& message);
};