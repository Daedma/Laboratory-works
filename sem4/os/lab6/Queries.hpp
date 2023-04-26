#pragma once
#include <SFML/System/Export.hpp>

struct Query
{
	enum Code : sf::Int8
	{
		NONE,
		USERNAME,
		DONE,
		ERROR
	};
};