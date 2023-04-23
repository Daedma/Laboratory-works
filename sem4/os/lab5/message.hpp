#include <string>
#include <chrono>
#include <memory>
#include <iostream>
#include <iterator>
#include <algorithm>

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

	bool operator<(const Message& rhs) const
	{
		return time < rhs.time;
	}

	void to_bytes(char* dest) const
	{
		std::memset(dest, 0, MAX_SIZE);
		std::memcpy(dest + OFFSET_TIME, &time, sizeof(std::chrono::time_point<std::chrono::system_clock>));
		std::memcpy(dest + OFFSET_USERNAME, user.data(), user.size());
		std::memcpy(dest + OFFSET_MESSAGE, content.data(), content.size());
	}

	void from_bytes(char* source)
	{
		std::memcpy(&time, source + OFFSET_TIME, sizeof(std::chrono::time_point<std::chrono::system_clock>));
		user = (std::string(source + OFFSET_USERNAME, MAX_USERNAME_SIZE));
		content = (std::string(source + OFFSET_MESSAGE, MAX_MESSAGE_SIZE));
	}

	struct less
	{
		bool operator()(const Message& lhs, const Message& rhs) const
		{
			return lhs < rhs;
		}
	};
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
		<< message.content.c_str();
	return os;
}
