#pragma once

#include <stdexcept>

#include "shape.hpp"

class sphere : public shape
{
	double m_radius;
public:
	sphere(double radius_)
	{
		radius(radius_);
	}

	void radius(double new_radius_)
	{
		if (new_radius_ <= 0.)
		{
			throw std::invalid_argument{ "radius must be positive" };
		}
		m_radius = new_radius_;
	}

	double radius() const noexcept
	{
		return m_radius;
	}

	vec_t normal(const vec_t& point) const override;

	void draw(LibBoard::Board& board, const LibBoard::Color &color) const override;

protected:
	std::vector<vec_t::value_type> intersection_points_Impl(const ray& ray_) const override;
};