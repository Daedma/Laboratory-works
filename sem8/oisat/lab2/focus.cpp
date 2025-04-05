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

double find_minimum(std::function<double(double)> func, double left_, double right_, double eps_, size_t iter_max_)
{
	double middle;
	for (size_t i = 0; i != iter_max_ && (right_ - left_) >= eps_; ++i)
	{
		middle = (right_ + left_) * .5;
		if (func(middle - eps_) > func(middle + eps_))
		{
			left_ = middle;
		}
		else
		{
			right_ = middle;
		}
	}
	return middle;
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

	return find_minimum(func, min_z, max_z_, PRECISION, ITER_MAX);
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

	return find_minimum(func, min_z, max_z_, PRECISION, ITER_MAX);
}

double calc_focus_distance_sdopl(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto raytraces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, 300, 2);

	auto func = [&raytraces, refr_ind_](double z_) {
		std::vector<vec_t> points;
		points.reserve(raytraces.size());

		std::vector<double> indieces = { 1., refr_ind_, 1. };

		plane p;
		p.shift({ 0., 0., z_ });

		for (const auto& raytrace : raytraces)
		{
			
		}

		};


	double min_z = lens_.minmax_z().first;
	double max_z = lens_.minmax_z().second + max_z_;

	return find_minimum(func, min_z, max_z_, PRECISION, ITER_MAX);
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
	double focus_sdxyol = calc_focus_distance_sdxyopl(lens, 1.5, 1000);

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
	board.drawDot(focus_sdxyol, 0);

	board.saveSVG("lens.svg");
	std::cout << "Lens illustration saved as lens.svg" << std::endl;

	return 0;
}