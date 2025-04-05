#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <functional>

#include <Board.h>
#include <glm/gtc/constants.hpp>

#include "ray.hpp"
#include "utils.hpp"
#include "biconvex_lens.hpp"
#include "plane.hpp"


namespace
{
	constexpr size_t RAYS_COUNT = 9;

	constexpr size_t ANGLES_COUNT = 100;

	constexpr double PRECISION = 1.e-6;

	constexpr size_t ITER_MAX = 1000;
}

std::pair<size_t, size_t> getFibonacciPairAbove(double val)
{
	size_t cur = 1, prev = 0;
	while (cur <= val)
	{
		cur += prev;
		prev = cur - prev;
	}
	return { cur, prev };
}

double find_minimum(std::function<double(double)> func, double left, double right, double eps)
{
	auto [fn, fnm1] = getFibonacciPairAbove((right - left) / eps);
	size_t fnm2 = fn - fnm1;
	double dx = right - left;
	double x1 = left + (fnm2 * dx) / fn;
	double x2 = left + (fnm1 * dx) / fn;
	double fx1 = func(x1);
	double fx2 = func(x2);
	fn = fnm1;
	fnm1 = fnm2;
	fnm2 = fn - fnm1;
	while (fn != fnm1)
	{
		if (fx1 < fx2)
		{
			right = x2;
			dx = right - left;
			x2 = x1;
			x1 = left + (fnm2 * dx) / fn;
			fx2 = fx1;
			fx1 = func(x1);
		}
		else
		{
			left = x1;
			dx = right - left;
			x1 = x2;
			x2 = left + (fnm1 * dx) / fn;
			fx1 = fx2;
			fx2 = func(x2);
		}
		fn = fnm1;
		fnm1 = fnm2;
		fnm2 = fn - fnm1;
	}
	return (x1 + x2) * .5;
}

double calc_optical_path_length(const std::vector<vec_t>& points, const std::vector<double>& indeces)
{
	double sum = 0.;
	for (size_t i = 0; i != points.size() - 1; ++i)
	{
		vec_t delta = points[i + 1] - points[i];
		sum += glm::length(delta) * indeces[i];
	}
	return sum;
}

vec_t mean(const std::vector<vec_t>& points_)
{
	vec_t sum = { 0., 0., 0. };
	for (const auto& point : points_)
	{
		sum += point;
	}
	return sum / static_cast<double>(points_.size());
}

double sd(const std::vector<vec_t>& points_, const vec_t& center_)
{
	double sum = 0.;
	for (const auto& point : points_)
	{
		sum += glm::dot(point - center_, point - center_);
	}
	return std::sqrt(sum / points_.size());
}

double sd(const std::vector<vec_t>& points)
{
	return sd(points, mean(points));
}


double calc_focus_distance_sdxy(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto raytraces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, 300, 2);

	auto func = [&raytraces](double z_) {
		std::vector<vec_t> intersections;
		intersections.reserve(raytraces.size());

		plane p;
		p.shift({ 0., 0., z_ });

		for (const auto& raytrace : raytraces)
		{
			intersections.emplace_back(p.intersection_points(raytrace.back()).front());
		}

		return sd(intersections, { 0., 0., z_ });
		};


	double min_z = lens_.minmax_z().first;
	double max_z = lens_.minmax_z().second + max_z_;

	return find_minimum(func, min_z, max_z, PRECISION);
}

double calc_focus_distance_sdxyopl(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto raytraces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, 300, 2);

	auto func = [&raytraces, refr_ind_](double z_) {
		std::vector<vec_t> intersections;
		intersections.reserve(raytraces.size());

		std::vector<double> indieces = { 1., refr_ind_, 1. };

		plane p;
		p.shift({ 0., 0., z_ });

		for (const auto& raytrace : raytraces)
		{
			vec_t intersection = p.intersection_points(raytrace.back()).front();

			std::vector<vec_t> points;
			points.reserve(raytrace.size() + 1);
			for (const auto& ray : raytrace)
			{
				points.emplace_back(ray.origin());
			}
			points.emplace_back(intersection);

			double optical_path_length = calc_optical_path_length(points, indieces);

			intersection.z = optical_path_length;
			intersections.emplace_back(intersection);
		}

		return sd(intersections);
		};


	double min_z = lens_.minmax_z().first;
	double max_z = lens_.minmax_z().second + max_z_;

	return find_minimum(func, min_z, max_z, PRECISION);
}

double calc_focus_distance_sdopl(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto raytraces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, 300, 2);

	auto get_equal_opl_points = [&raytraces, refr_ind_](double h_) {
		std::vector<vec_t> points;
		points.reserve(raytraces.size());

		std::vector<double> indieces = { 1., refr_ind_ };

		for (const auto& raytrace : raytraces)
		{
			double opl1 = glm::distance(raytrace[0].origin(), raytrace[1].origin());
			double opl2 = glm::distance(raytrace[1].origin(), raytrace[2].origin()) * refr_ind_;
			double l3 = h_ - opl1 - opl2;
			vec_t r = raytrace[2].origin() + raytrace[2].direction() * l3;
			points.emplace_back(r);
		}

		return points;
		};

	auto func = [&get_equal_opl_points](double h_) {
		return sd(get_equal_opl_points(h_));
		};


	double min_h = lens_.minmax_z().first;
	double max_h = lens_.minmax_z().second + max_z_;

	double h = find_minimum(func, min_h, max_h, PRECISION);

	return mean(get_equal_opl_points(h)).z;
}

int main()
{
	LibBoard::Board board;
	board.setFillColor(LibBoard::Color::White);
	board.setPenColor(LibBoard::Color::White);
	board.setLineWidth(2.);
	board.fillRectangle(-1000, -1000, 2000, 2000);
	board.moveCenter({ 0., 0. });

	double lens_offset = 50;
	biconvex_lens lens(80, 300, lens_offset, 100, 300, lens_offset);

	double focus_sd = calc_focus_distance_sdxy(lens, 1.5, 1000);
	double focus_sdxyopl = calc_focus_distance_sdxyopl(lens, 1.5, 1000);
	double focus_sdopl = calc_focus_distance_sdopl(lens, 1.5, 1000);

	lens.draw(board, LibBoard::Color::DarkCyan);

	auto raytraces = trace_rays_through_lens(lens, 1., 1.5, 9, 500, 2);

	std::array<LibBoard::Color, 2> colors = {
		LibBoard::Color::Red,
		LibBoard::Color::Green
	};

	for (const auto& raytrace : raytraces)
	{
		for (size_t i = 0; i != raytrace.size() - 1; ++i)
		{
			LibBoard::Color color = colors[i % colors.size()];
			vec_t end = raytrace[i + 1].origin();
			raytrace[i].draw(board, color, end);
		}
		raytrace.back().draw(board, LibBoard::Color::Blue, 500);
	}

	board.setLineWidth(10.);

	board.setPenColor(LibBoard::Color::Red);
	board.drawDot(focus_sd, 0);

	board.setPenColor(LibBoard::Color::Green);
	board.drawDot(focus_sdxyopl, 0);

	board.setPenColor(LibBoard::Color::Blue);
	board.drawDot(focus_sdopl, 0);

	board.saveSVG("lens.svg");
	std::cout << "Lens illustration saved as lens.svg" << std::endl;

	return 0;
}