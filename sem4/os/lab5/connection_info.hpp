#include <WinSock2.h>

namespace connection_info
{
	inline constexpr const char* ADDRESS = "127.0.0.1";

	inline constexpr short SERVER_PORT = 1488;

	inline constexpr short CLIENT_PORT = 1337;

	inline constexpr int MAX_QUEUE_SIZE = SOMAXCONN;

	inline constexpr const char* SEMAPHORE_NAME = "client count in chat";
};