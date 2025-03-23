#include <glm/gtc/matrix_transform.hpp>

#include "ray.hpp"
#include "shape.hpp"

std::vector<vec_t> shape::intersection_points(const ray& ray_) const
{
	ray fiting_ray = ray_.apply_transform(m_transform);
	std::vector<vec_t::value_type> lenghts = intersection_points_Impl(fiting_ray);
	std::vector<vec_t> result;
	for (const auto& lenght : lenghts)
	{
		if (lenght >= 0)
		{
			result.emplace_back(ray_.origin() + ray_.direction() * lenght);
		}
	}
	return result;
}

std::vector<ray> shape::reflect_ray(const ray& ray_) const
{
	std::vector<vec_t> origins = intersection_points(ray_);
	std::vector<ray> reflected;
	for (const auto& origin : origins)
	{
		vec_t normal_ = normal(origin);
		vec_t direction = ray_.direction() - 2 * glm::dot(ray_.direction(), normal_) * normal_;
		reflected.emplace_back(origin, direction);
	}
	return reflected;
}

std::vector<ray> shape::refract_ray(const ray& ray_, double refr_ind_out, double refr_ind_in) const
{
	std::vector<vec_t> origins = intersection_points(ray_);
	std::vector<ray> refracted;
	bool inside = false;

	for (const auto& origin : origins)
	{
		vec_t normal_ = normal(origin);
		double refr_ratio = inside ? refr_ind_in / refr_ind_out : refr_ind_out / refr_ind_in;
		vec_t incident = ray_.direction();
		double cos_i = -glm::dot(incident, normal_);
		double sin_t2 = refr_ratio * refr_ratio * (1.0 - cos_i * cos_i);

		if (sin_t2 > 1.0)
		{
			vec_t direction = incident + 2 * cos_i * normal_;
			refracted.emplace_back(origin, direction);
		}
		else
		{
			double cos_t = glm::sqrt(1.0 - sin_t2);
			vec_t direction = refr_ratio * incident + (refr_ratio * cos_i - cos_t) * normal_;
			refracted.emplace_back(origin, direction);
		}

		inside = !inside;
	}

	return refracted;
}

void shape::update_transform() noexcept
{
	m_transform = mat_t(1);
	m_transform = glm::translate(m_transform, m_offset);
	m_transform = glm::rotate(m_transform, m_rotation.z, { 0, 0, 1 });
	m_transform = glm::rotate(m_transform, m_rotation.y, { 0, 1, 0 });
	m_transform = glm::rotate(m_transform, m_rotation.x, { 1, 0, 0 });
}