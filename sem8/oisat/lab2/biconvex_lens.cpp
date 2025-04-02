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
	auto init_style = board.style();

	board.setFillColorRGBf(1., 1., 1., 0.);
	m_surface_1.draw(board, color);
	m_surface_2.draw(board, color);
	board.setFillColor(init_style.fillColor);

	double ld = board.style().lineWidth;

	double left_1 = -m_surface_1.z_radius() + m_surface_1.shift().z - ld;
	double right_1 = m_surface_1.z_radius() + m_surface_1.shift().z + ld;

	double left_2 = -m_surface_2.z_radius() + m_surface_2.shift().z - ld;
	double right_2 = m_surface_2.z_radius() + m_surface_2.shift().z + ld;


	std::array<LibBoard::Point, 4> inner_frame = { {
		{ left_1, m_radius },
		{ right_2, m_radius },
		{ right_2, -m_radius },
		{ left_1, -m_radius }
		} };

	double max_radius = std::max(m_surface_1.y_radius(), m_surface_2.y_radius()) + ld;

	double left = std::min(left_1, left_2);
	double right = std::max(right_1, right_2);

	std::array<LibBoard::Point, 4> outer_frame = { {
		{ left, max_radius },
		{ right, max_radius },
		{ right, -max_radius },
		{ left, -max_radius }
		} };

	// board.setPenColor(LibBoard::Color::Red);
	// board.drawRectangle(
	// 	outer_frame[0].x, outer_frame[0].y,
	// 	abs(outer_frame[1].x - outer_frame[0].x),
	// 	abs(outer_frame[3].y - outer_frame[0].y)
	// );

	// board.setPenColor(LibBoard::Color::Blue);
	// board.drawRectangle(
	// 	inner_frame[0].x, inner_frame[0].y,
	// 	abs(inner_frame[1].x - inner_frame[0].x),
	// 	abs(inner_frame[3].y - inner_frame[0].y)
	// );
	LibBoard::Polyline frame(LibBoard::Path::Closed, board.style());
	frame.setLineWidth(0.);
	frame << outer_frame[0] << outer_frame[1] << outer_frame[2] << outer_frame[3];

	LibBoard::Path hole;
	hole << inner_frame[0] << inner_frame[1] << inner_frame[2] << inner_frame[3];

	frame.addHole(hole);

	board.drawRectangle(m_cuts.first, m_radius, m_cuts.second - m_cuts.first, 2 * m_radius);
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
