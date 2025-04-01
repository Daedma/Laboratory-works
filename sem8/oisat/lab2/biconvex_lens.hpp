#pragma once

#include <utility>
#include <algorithm>
#include <vector>

#include "ellipse.hpp"

class biconvex_lens : public shape
{
	ellipse m_surface_1;

	ellipse m_surface_2;

public:
	biconvex_lens(const ellipse& surface_1_, double center_1_, const ellipse& surface_2_, double center_2_) :
		m_surface_1(surface_1_), m_surface_2(surface_2_)
	{
		m_surface_1.shift(vec_t{ 0, 0, center_1_ - m_surface_1.z_radius() });
		m_surface_1.rotation((vec_t(0)));
		m_surface_2.shift(vec_t{ 0, 0, -center_2_ + m_surface_2.z_radius() });
		m_surface_2.rotation((vec_t(0)));
	}

	biconvex_lens(double a1_, double b1_, double a2_, double b2_, double center_1_, double center_2_) :
		biconvex_lens({ b1_, b1_, a1_ }, center_1_, { b2_, b2_, a2_ }, center_2_)
	{}

	vec_t normal(const vec_t& point) const override;

	void draw(LibBoard::Board& board, const LibBoard::Color& color) const override;

protected:
	std::vector<vec_t::value_type> intersection_points_Impl(const ray& ray_) const override;

private:
	std::pair<double, double> main_planes_z() const noexcept
	{
		const double a1 = m_surface_1.x_radius();
		const double c1 = m_surface_1.z_radius();
		const double r1 = m_surface_1.shift().z;
		const double a2 = m_surface_2.x_radius();
		const double c2 = m_surface_2.z_radius();
		const double r2 = m_surface_2.shift().z;

		const double a = a2 * a2 / (c2 * c2) - a1 * a1 / (c1 * c1);
		const double b = 2 * (a1 * a1 * r1 / (c1 * c1) - a2 * a2 * r2 / (c2 * c2));
		const double c = a1 * a1 - a2 * a2 + a2 * a2 * r2 * r2 / (c2 * c2) - a1 * a1 * r1 * r1 / (c1 * c1);

		const double disc = b * b - 4. * a * c;

		std::vector<double> radiuses = { a1, a2 };

		if (std::abs(disc) <= tolerance)
		{
			const double z = -b / (2 * a);
			const double r = a1 * std::sqrt(1 - (z - r1) * (z - r1) / (c1 * c1));
			radiuses.emplace_back(r);
		}
		else if (disc > 0)
		{
			const double disc_root = std::sqrt(disc);

			const double z1 = (-b - disc_root) / (2 * a);
			const double radius1 = a1 * std::sqrt(1 - (z1 - r1) * (z1 - r1) / (c1 * c1));
			radiuses.emplace_back(radius1);

			const double z2 = (-b + disc_root) / (2 * a);
			const double radius2 = a1 * std::sqrt(1 - (z2 - r1) * (z2 - r1) / (c1 * c1));
			radiuses.emplace_back(radius2);
		}

		const double radius = *std::min_element(radiuses.cbegin(), radiuses.cend());

		const double z1 = -c1 * std::sqrt(1 - radius * radius / (a1 * a1)) + r1;
		const double z2 = c2 * std::sqrt(1 - radius * radius / (a2 * a2)) + r2;

		return std::minmax(z1, z2);
	}
};