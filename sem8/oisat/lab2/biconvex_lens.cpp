#include <Board.h>

#include <cmath>
#include <array>

#include "biconvex_lens.hpp"

vec_t biconvex_lens::normal(const vec_t& point) const
{
	if (point.z < m_cuts.first)
	{
		return m_surface_1.normal(point);
	}
	else if (point.z > m_cuts.second)
	{
		return m_surface_2.normal(point);
	}
	else
	{
		vec_t normal_ = point;
		normal_.z = 0;
		return glm::normalize(normal_);
	}
}

void biconvex_lens::draw(LibBoard::Board& board, const LibBoard::Color& color) const
{
	m_surface_1.draw(board, color);
	m_surface_2.draw(board, color);

	double center_1 = m_surface_1.z_radius() + m_surface_1.shift().z;
	double center_2 = m_surface_2.z_radius() - m_surface_2.shift().z;

	std::array<LibBoard::Point, 4> inner_frame = { {
		{ center_1, m_radius },
		{ center_2, m_radius },
		{ center_2, -m_radius },
		{ center_1, -m_radius }
		} };

	center_1 += 2 * m_surface_1.z_radius();
	center_2 -= 2 * m_surface_2.z_radius();

	double max_radius = std::max(m_surface_1.y_radius(), m_surface_2.y_radius());

	std::array<LibBoard::Point, 4> outer_frame = { {
		{ center_1, max_radius },
		{ center_2, max_radius },
		{ center_2, -max_radius },
		{ center_1, -max_radius }
		} };

	LibBoard::Polyline frame(LibBoard::Path::Closed, board.style());
	frame << outer_frame[0] << outer_frame[1] << outer_frame[2] << outer_frame[3];

	LibBoard::Path hole;
	hole << inner_frame[0] << inner_frame[1] << inner_frame[2] << inner_frame[3];

	frame.addHole(hole);

	board << frame;
}

std::vector<vec_t::value_type> biconvex_lens::intersection_points_Impl(const ray& ray_) const
{
	std::vector<vec_t::value_type> lengths;

	auto points1 = m_surface_1.intersection_points(ray_);
	for (const auto& point : points1)
	{
		if (point.z < m_cuts.first)
		{
			lengths.emplace_back(glm::length(point));
		}
	}

	auto points2 = m_surface_2.intersection_points(ray_);
	for (const auto& point : points2)
	{
		if (point.z >= m_cuts.second)
		{
			lengths.emplace_back(glm::length(point));
		}
	}

	return lengths;
}
