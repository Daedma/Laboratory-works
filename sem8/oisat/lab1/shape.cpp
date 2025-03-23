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
		result.emplace_back(ray_.origin() + ray_.direction() * lenght);
	}
	return result;
}

void shape::update_transform() noexcept
{
	m_transform = mat_t(1);
	m_transform = glm::translate(m_transform, m_offset);
	m_transform = glm::rotate(m_transform, m_rotation.z, { 0, 0, 1 });
	m_transform = glm::rotate(m_transform, m_rotation.y, { 0, 1, 0 });
	m_transform = glm::rotate(m_transform, m_rotation.x, { 1, 0, 0 });
	m_transform = glm::scale(m_transform, m_scale);
}