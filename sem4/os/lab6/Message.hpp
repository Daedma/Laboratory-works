#pragma once

#include <chrono>
#include <string>
#include <SFML/Network/Packet.hpp>
#include <memory>
#include <format>

struct Message
{
	using clock_t = std::chrono::system_clock;
	using time_point = std::chrono::time_point<std::chrono::system_clock>;

	time_point time;
	std::string user;
	std::string content;

	std::string toString() const
	{
		std::string result;
		auto beginOfDay = std::chrono::floor<std::chrono::days>(time);
		auto clockTime = std::chrono::hh_mm_ss(time - beginOfDay);
		result = std::format("[{:%T}] {} : {}", clockTime, user, content);
		return result;
	}

	struct less
	{
		bool operator()(const Message& lhs, const Message& rhs) const
		{
			return lhs.time < rhs.time;
		}
	};
};

static sf::Packet& operator>>(sf::Packet& pack, Message& mess)
{
	char buff[sizeof(Message::time_point)];
	pack >> buff;
	std::memcpy(&mess.time, buff, sizeof(buff));
	pack >> mess.user >> mess.content;
	return pack;
}

static sf::Packet& operator<<(sf::Packet& pack, const Message& mess)
{
	char buff[sizeof(Message::time_point)];
	std::memcpy(buff, &mess.time, sizeof(buff));
	return pack << buff << mess.user << mess.content;
}