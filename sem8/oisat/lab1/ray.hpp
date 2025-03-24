#pragma once

#include <stdexcept>
#include <Board.h>

#include "types.hpp"

class ray
{
	vec_t m_origin;

	vec_t m_direction;

public:
	ray(const vec_t& origin_, const vec_t& direction_) :
		m_origin(origin_)
	{
		direction(direction_);
	}

	void origin(const vec_t& new_origin_)
	{
		m_origin = new_origin_;
	}

	const vec_t& origin() const noexcept
	{
		return m_origin;
	}

	void direction(const vec_t& new_direction_)
	{
		vec_t::value_type length = glm::length(new_direction_);
		if (length == 0.)
		{
			throw std::invalid_argument{ "direction must be non-zero vector" };
		}
		m_direction = new_direction_ / length;
	}

	const vec_t& direction() const noexcept
	{
		return m_direction;
	}

	ray apply_transform(const mat_t& transform) const noexcept
	{
		auto origin_res = glm::vec<vec_t::length() + 1, vec_t::value_type>{ m_origin, 1 };
		auto direction_res = glm::vec<vec_t::length() + 1, vec_t::value_type>{ m_direction, 0 };
		origin_res = transform * origin_res;
		direction_res = transform * direction_res;
		return { { origin_res }, { direction_res } };
	}

    void draw(LibBoard::Board& board, const LibBoard::Color& color, const vec_t& end) const
    {
        board.setPenColor(color);
        board.drawArrow(m_origin.x, m_origin.y, end.x, end.y, LibBoard::Arrow::ExtremityType::Stick);
    }
};