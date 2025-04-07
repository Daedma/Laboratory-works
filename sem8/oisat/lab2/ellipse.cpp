#include <cmath>
#include <Board.h>

#include "ray.hpp"
#include "ellipse.hpp"

std::vector<vec_t::value_type> ellipse::intersection_points_Impl(const ray& ray_) const
{
	std::vector<vec_t::value_type> points;

	vec_t::value_type a = (ray_.direction().x * ray_.direction().x) / (m_x_radius * m_x_radius) +
		(ray_.direction().y * ray_.direction().y) / (m_y_radius * m_y_radius) +
		(ray_.direction().z * ray_.direction().z) / (m_z_radius * m_z_radius);

	vec_t::value_type b = 2 * ((ray_.origin().x * ray_.direction().x) / (m_x_radius * m_x_radius) +
		(ray_.origin().y * ray_.direction().y) / (m_y_radius * m_y_radius) +
		(ray_.origin().z * ray_.direction().z) / (m_z_radius * m_z_radius));

	vec_t::value_type c = (ray_.origin().x * ray_.origin().x) / (m_x_radius * m_x_radius) +
		(ray_.origin().y * ray_.origin().y) / (m_y_radius * m_y_radius) +
		(ray_.origin().z * ray_.origin().z) / (m_z_radius * m_z_radius) - 1;

	vec_t::value_type disc = b * b - 4 * a * c;

	if (std::abs(disc) <= tolerance)
	{
		points.emplace_back(-b / (2 * a));
	}
	else if (disc > 0)
	{
		vec_t::value_type disc_root = std::sqrt(disc);
		points.emplace_back((-b - disc_root) / (2 * a));
		points.emplace_back((-b + disc_root) / (2 * a));
	}

	return points;
}

vec_t ellipse::normal(const vec_t& point) const
{
	vec_t normal;
	normal.x = 2 * (point.x - shift().x) / (m_x_radius * m_x_radius);
	normal.y = 2 * (point.y - shift().y) / (m_y_radius * m_y_radius);
	normal.z = 2 * (point.z - shift().z) / (m_z_radius * m_z_radius);
	return glm::normalize(normal);
}

void ellipse::draw(LibBoard::Board& board, const LibBoard::Color &color) const
{
	board.setPenColor(color);
	board.drawEllipse(shift().z, shift().y, m_z_radius, m_y_radius);
}

