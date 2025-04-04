#pragma once

#include "shape.hpp"

class plane : public shape
{
public:
	vec_t normal(const vec_t& point) const override;

	void draw(LibBoard::Board& board, const LibBoard::Color &color) const override;

protected:
	std::vector<vec_t::value_type> intersection_points_Impl(const ray& ray_) const override;
};