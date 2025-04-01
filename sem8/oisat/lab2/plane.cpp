#include <Board.h>

#include "ray.hpp"
#include "plane.hpp"

std::vector<vec_t::value_type> plane::intersection_points_Impl(const ray& ray_) const
{
	std::vector<vec_t::value_type> points;

	if (ray_.direction().z != 0)
	{
		points.emplace_back(-ray_.origin().z / ray_.direction().z);
	}

	return points;
}

vec_t plane::normal(const vec_t& point) const
{
	vec_t result(transform()[0].z, transform()[1].z, transform()[2].z);
	return glm::normalize(result);
}

void plane::draw(LibBoard::Board& board, const LibBoard::Color &color) const
{
	board.setPenColor(color);
	auto aabb = board.boundingBox(LibBoard::LineWidthFlag::UseLineWidth);
	board.drawLine(aabb.left, shift().y, aabb.right(), shift().y);
}
