#include <string>
#include <chrono>
#include <memory>
#include <iostream>

struct Message
{
	std::chrono::time_point<std::chrono::system_clock> time;

	std::string user;

	std::string content;

	inline static constexpr size_t MAX_MESSAGE_SIZE = 1024;

	inline static constexpr size_t MAX_USERNAME_SIZE = 8;

	inline static constexpr size_t MAX_SIZE = MAX_MESSAGE_SIZE +
		MAX_USERNAME_SIZE + sizeof(std::chrono::time_point<std::chrono::system_clock>);

	inline static constexpr size_t OFFSET_TIME = 0;

	inline static constexpr size_t OFFSET_USERNAME = OFFSET_TIME + sizeof(std::chrono::time_point<std::chrono::system_clock>);

	inline static constexpr size_t OFFSET_MESSAGE = OFFSET_USERNAME + MAX_USERNAME_SIZE;

	bool operator<(const Message& rhs)
	{
		return time < rhs.time;
	}

	void to_bytes(char* dest)
	{
		std::memset(dest, 0, MAX_SIZE);
		std::memcpy(dest + OFFSET_TIME, &time, sizeof(std::chrono::time_point<std::chrono::system_clock>));
		std::memcpy(dest + OFFSET_USERNAME, user.data(), user.size() + 1);
		std::memcpy(dest + OFFSET_MESSAGE, content.data(), content.size() + 1);
	}

	void from_bytes(char* source)
	{
		std::memcpy(&time, source + OFFSET_TIME, sizeof(std::chrono::time_point<std::chrono::system_clock>));
		user = source + OFFSET_USERNAME;
		content = source + OFFSET_MESSAGE;
	}

};

std::ostream& operator<<(std::ostream& os, const Message& message)
{
	using namespace std::chrono;
	auto d = message.time.time_since_epoch();
	const auto hrs = duration_cast<hours>(d);
	const auto mins = duration_cast<minutes>(d - hrs);
	const auto secs = duration_cast<seconds>(d - hrs - mins);
	os << "[" << hrs.count() << ":" << mins.count()
		<< ":" << secs.count() << "] " << message.user << " : "
		<< message.content;
	return os;
}