#pragma once

#include <vector>
#include <optional>

#include "types.hpp"

class ray;
namespace LibBoard
{
	class Board;
	class Color;
}

class shape
{
	vec_t m_offset = vec_t{ 0, 0, 0 };

	vec_t m_rotation = vec_t{ 0, 0, 0 };

	mat_t m_transform = (mat_t(1));

public:
	virtual ~shape() = default;

	void shift(const vec_t& new_offset_) noexcept
	{
		m_offset = new_offset_;
		update_transform();
	}

	const vec_t& shift() const noexcept
	{
		return m_offset;
	}

	void rotation(const vec_t& new_rotation_) noexcept
	{
		m_rotation = new_rotation_;
		update_transform();
	}

	const vec_t& rotation() const noexcept
	{
		return m_rotation;
	}

	const mat_t& transform() const noexcept
	{
		return m_transform;
	}

	std::vector<vec_t> intersection_points(const ray& ray_) const;

	std::optional<ray> reflect_ray(const ray& ray_) const;

	std::optional<ray> refract_ray(const ray& ray_, double refr_ind_out, double refr_ind_in) const;

	virtual vec_t normal(const vec_t& point) const = 0;

	virtual void draw(LibBoard::Board& board, const LibBoard::Color &color) const = 0;

protected:
	virtual std::vector<vec_t::value_type> intersection_points_Impl(const ray& ray_) const = 0;

private:
	void update_transform() noexcept;
};