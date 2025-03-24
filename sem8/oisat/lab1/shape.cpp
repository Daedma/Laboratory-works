#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

#include "ray.hpp"
#include "shape.hpp"

std::vector<vec_t> shape::intersection_points(const ray& ray_) const
{
	ray fiting_ray = ray_.apply_transform(m_transform);
	std::vector<vec_t::value_type> lenghts = intersection_points_Impl(fiting_ray);
	std::vector<vec_t> result;
	for (const auto& lenght : lenghts)
	{
		if (lenght > 0)
		{
			result.emplace_back(ray_.origin() + ray_.direction() * lenght);
		}
	}
	return result;
}

std::optional<ray> shape::reflect_ray(const ray& ray_) const
{
	std::vector<vec_t> origins = intersection_points(ray_);
	if (!origins.empty())
	{
		auto closest_origin = *std::min_element(origins.begin(), origins.end(),
			[&ray_](const vec_t& a, const vec_t& b) {
				return glm::length(a - ray_.origin()) < glm::length(b - ray_.origin());
			});

		vec_t normal_ = normal(closest_origin);
		vec_t direction = ray_.direction() - 2 * glm::dot(ray_.direction(), normal_) * normal_;
		return ray(closest_origin, direction);
	}
	return std::nullopt;
}

std::optional<ray> shape::refract_ray(const ray& ray_, double refr_ind_out, double refr_ind_in) const
{
	std::vector<vec_t> origins = intersection_points(ray_);

	if (!origins.empty())
	{
		auto closest_origin = *std::min_element(origins.begin(), origins.end(),
			[&ray_](const vec_t& a, const vec_t& b) {
				return glm::length(a - ray_.origin()) < glm::length(b - ray_.origin());
			});

		vec_t normal_ = normal(closest_origin);
		vec_t incident = ray_.direction();
		double n1 = refr_ind_out;
		double n2 = refr_ind_in;
		double refr_ratio = n1 / n2;
		double cos_i = -glm::dot(incident, normal_);
		double sub_root = (n2*n2 - n1*n1) / (cos_i * cos_i * n1 * n1) + 1.;

		if (sub_root < 0.)
		{
			vec_t direction = incident + 2 * cos_i * normal_;
			return ray(closest_origin, direction);
		}
		else
		{
			double cos_t = std::sqrt(sub_root);
			vec_t direction = refr_ratio * incident - refr_ratio * cos_i * normal_ * (1. - cos_t);
			return ray(closest_origin, direction);
		}

	}

	return std::nullopt;
}

void shape::update_transform() noexcept
{
	m_transform = mat_t(1);
	m_transform = glm::translate(m_transform, m_offset);
	m_transform = glm::rotate(m_transform, m_rotation.z, { 0, 0, 1 });
	m_transform = glm::rotate(m_transform, m_rotation.y, { 0, 1, 0 });
	m_transform = glm::rotate(m_transform, m_rotation.x, { 1, 0, 0 });
}