#include <Board.h>

#include <cmath>
#include <array>

#include "ray.hpp"
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

	LibBoard::Polyline frame(LibBoard::Path::Closed, board.style());
	frame.setLineWidth(0.);
	frame << outer_frame[0] << outer_frame[1] << outer_frame[2] << outer_frame[3];

	LibBoard::Path hole;
	hole << inner_frame[0] << inner_frame[1] << inner_frame[2] << inner_frame[3];

	frame.addHole(hole);
	board << frame;

	board.setPenColor(color);
	board.drawRectangle(m_cuts.first, m_radius, m_cuts.second - m_cuts.first, 2 * m_radius);
}

std::vector<vec_t::value_type> biconvex_lens::intersection_points_Impl(const ray& ray_) const
{
	std::vector<vec_t::value_type> lengths;

	auto points1 = m_surface_1.intersection_points(ray_);
	for (const auto& point : points1)
	{
		if (point.z < m_cuts.first)
		{
			lengths.emplace_back(glm::distance(point, ray_.origin()));
		}
	}
	
	auto points2 = m_surface_2.intersection_points(ray_);
	for (const auto& point : points2)
	{
		if (point.z >= m_cuts.second)
		{
			lengths.emplace_back(glm::distance(point, ray_.origin()));
		}
	}

	return lengths;
}

std::vector<std::vector<ray>> trace_rays_through_lens(const biconvex_lens& lens_, double refr_ind_out_,
	double refr_ind_in_, const std::vector<ray>& input_rays_, size_t max_refractions_)
{
	std::vector<std::vector<ray>> raytraces(input_rays_.size());
	for (size_t i = 0; i != input_rays_.size(); ++i)
	{
		raytraces[i].emplace_back(input_rays_[i]);
	}

	for (auto& raytrace : raytraces)
	{
		double refr_ind_in = refr_ind_in_;
		double refr_ind_out = refr_ind_out_;
		for (size_t i = 0; i != raytrace.size() && i != max_refractions_; ++i)
		{
			const ray& falling_ray = raytrace[i];
			auto refracted_ray = lens_.refract_ray(falling_ray, refr_ind_out, refr_ind_in);

			if (refracted_ray.has_value())
			{
				raytrace.emplace_back(refracted_ray.value());

				bool is_reflected = glm::dot(refracted_ray->direction(), falling_ray.direction()) < 0;
				if (!is_reflected)
				{
					std::swap(refr_ind_in, refr_ind_out);
				}
			}
		}
	}

	return raytraces;
}

std::vector<std::vector<ray>> trace_rays_through_lens(const biconvex_lens& lens_, double refr_ind_in,
	double refr_ind_out, size_t num_rays_, double distance_, size_t max_refractions_)
{
	std::vector<ray> input_rays;

	const double z = lens_.minmax_z().first - distance_;
	const double x = 0.;

	double min_y = lens_.minmax_y().first / 4;
	double max_y = lens_.minmax_y().second / 4;
	const double step = (max_y - min_y) / (num_rays_ + 1);
	for (size_t i = 0; i != num_rays_; ++i)
	{
		double y = min_y + (i + 1) * step;
		input_rays.emplace_back(vec_t{ x, y, z }, vec_t{ 0., 0., 1. });
	}

	return trace_rays_through_lens(lens_, refr_ind_in, refr_ind_out, input_rays, max_refractions_);
}
