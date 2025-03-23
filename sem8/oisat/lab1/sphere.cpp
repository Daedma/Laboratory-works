#include <cmath>

#include "ray.hpp"
#include "sphere.hpp"

std::vector<vec_t::value_type> sphere::intersection_points_Impl(const ray& ray_) const
{
	std::vector<vec_t::value_type> points;

	vec_t::value_type b = glm::dot(ray_.origin(), ray_.direction());
	vec_t::value_type ac = glm::dot(ray_.origin(), ray_.origin()) - m_radius * m_radius;
	vec_t::value_type disc = b * b - ac;

	if (std::abs(disc) <= tolerance)
	{
		points.emplace_back(-b);
	}
	else if (disc > 0)
	{
		vec_t::value_type disc_root = std::sqrt(disc);
		points.emplace_back(-b + disc_root);
		points.emplace_back(-b - disc_root);
	}

	return points;
}

vec_t sphere::normal(const vec_t& point) const
{
	return glm::normalize(point - shift());
}
