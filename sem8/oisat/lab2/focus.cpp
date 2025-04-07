#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <limits>

#include <Board.h>
#include <glm/gtc/constants.hpp>

#include "ray.hpp"
#include "utils.hpp"
#include "biconvex_lens.hpp"
#include "plane.hpp"


namespace
{
	constexpr size_t RAYS_COUNT = 9;

	constexpr double PRECISION = 1.e-6;

	constexpr double RAYS_DISTANCE = 2.;

	constexpr size_t MAX_ITERATIONS = 100;
}

double minimum(std::function<double(double)> func, double left, double right, double eps, size_t iterMax = MAX_ITERATIONS)
{
	constexpr double REVERSE_PHI = 0.6180339887498948;
	double dx = right - left;
	double x1 = right - dx * REVERSE_PHI, x2 = left + dx * REVERSE_PHI;
	double fx1 = func(x1);
	double fx2 = func(x2);

	for (size_t i = 0; i != iterMax && (x2 - x1) >= eps; ++i)
	{
		if (fx1 >= fx2)
		{
			left = x1;
			dx = right - left;
			x1 = x2;
			x2 = left + dx * REVERSE_PHI;
			fx1 = fx2;
			fx2 = func(x2);
		}
		else
		{
			right = x2;
			dx = right - left;
			x2 = x1;
			x1 = right - dx * REVERSE_PHI;
			fx2 = fx1;
			fx1 = func(x1);
		}
	}
	return (x2 + x1) * 0.5;
}

double opl(const std::vector<vec_t>& points, const std::vector<double>& indices)
{
	double sum = 0.;
	for (size_t i = 0; i != points.size() - 1; ++i)
	{
		vec_t delta = points[i + 1] - points[i];
		sum += glm::length(delta) * indices[i];
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
	if (points_.empty())
	{
		return std::numeric_limits<double>::max();
	}

	return std::sqrt(sum / points_.size());
}

double sd(const std::vector<vec_t>& points)
{
	return sd(points, mean(points));
}

// Find focus of a lens using sco of intersection points minimization
double focus_ps(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto ray_traces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, RAYS_DISTANCE, 2);

	auto intersection_points_sd = [&ray_traces](double z_) {
		std::vector<vec_t> intersections;
		intersections.reserve(ray_traces.size());

		plane p;
		p.shift({ 0., 0., z_ });

		for (const auto& ray_trace : ray_traces)
		{
			std::vector<vec_t> intersection = p.intersection_points(ray_trace.back());
			if(!intersection.empty())
			{
				intersections.emplace_back(p.intersection_points(ray_trace.back()).front());
			}
		}
		return sd(intersections, { 0., 0., z_ });
		};


	double min_z = lens_.minmax_z().second;
	double max_z = lens_.minmax_z().second + max_z_;

	return minimum(intersection_points_sd, min_z, max_z, PRECISION);
}

// Find focus of a lens using sco of intersection points with optical path length minimization
double focus_popls(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto ray_traces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, RAYS_DISTANCE, 2);

	auto popl_sd = [&ray_traces, refr_ind_](double z_) {
		std::vector<vec_t> intersections;
		intersections.reserve(ray_traces.size());

		std::vector<double> indices = { 1., refr_ind_, 1. };

		plane p;
		p.shift({ 0., 0., z_ });

		for (const auto& ray_trace : ray_traces)
		{
			vec_t intersection = p.intersection_points(ray_trace.back()).front();

			std::vector<vec_t> points;
			points.reserve(ray_trace.size() + 1);
			for (const auto& ray : ray_trace)
			{
				points.emplace_back(ray.origin());
			}
			points.emplace_back(intersection);

			double optical_path_length = opl(points, indices);

			intersection.z = optical_path_length;
			intersections.emplace_back(intersection);
		}

		return sd(intersections);
		};


	double min_z = lens_.minmax_z().second;
	double max_z = lens_.minmax_z().second + max_z_;

	return minimum(popl_sd, min_z, max_z, PRECISION);
}

// Find focus of a lens using sco of consistent optical path length minimization
double focus_copls(const biconvex_lens& lens_, double refr_ind_, double max_z_)
{
	auto ray_traces = trace_rays_through_lens(lens_, 1., refr_ind_, RAYS_COUNT, RAYS_DISTANCE, 2);

	auto copl_points = [&ray_traces, refr_ind_](double h_) {
		std::vector<vec_t> points;
		points.reserve(ray_traces.size());

		std::vector<double> indices = { 1., refr_ind_ };

		for (const auto& ray_trace : ray_traces)
		{
			double opl1 = glm::distance(ray_trace[0].origin(), ray_trace[1].origin());
			double opl2 = glm::distance(ray_trace[1].origin(), ray_trace[2].origin()) * refr_ind_;
			double l3 = h_ - opl1 - opl2;
			vec_t r = ray_trace[2].origin() + ray_trace[2].direction() * l3;
			points.emplace_back(r);
		}

		return points;
		};

	auto func = [&copl_points](double h_) {
		return sd(copl_points(h_));
		};


	double min_h = lens_.minmax_z().second;
	double max_h = lens_.minmax_z().second + max_z_;

	double h = minimum(func, min_h, max_h, PRECISION);

	return mean(copl_points(h)).z;
}

int main()
{
	LibBoard::Board board;
	board.setFillColor(LibBoard::Color::White);
	board.setPenColor(LibBoard::Color::White);
	board.setLineWidth(0.02);
	board.moveCenter({ 0., 0. });

	double a1, a2;
	double b1, b2;
	double center_1, center_2;
	double refr_ind;

	std::cout << "a - radius of ellipse x and y; b - radius of ellipse z;" << std::endl;
	std::cout << "The centers of elliptical surfaces are positive and denote a shift from the origin." << std::endl;
	std::cout << "(a1, b1) > ";
	std::cin >> a1 >> b1;
	std::cout << "(a2, b2) > ";
	std::cin >> a2 >> b2;
	std::cout << "centers of elliptical surfaces (1, 2) > ";
	std::cin >> center_1 >> center_2;
	std::cout << "refractive index of lens> ";
	std::cin >> refr_ind;

	biconvex_lens lens(a1, b1, center_1, a2, b2, center_2);

	double fps = focus_ps(lens, refr_ind, 20);
	double fpopls = focus_popls(lens, refr_ind, 20);
	double fcopls = focus_copls(lens, refr_ind, 20);

	std::cout << "Focus (sd of points) : " << fps << " (red dot)" << std::endl;
	std::cout << "Focus (sd of points and optical path length) : " << fpopls << " (green dot)" << std::endl;
	std::cout << "Focus (sd of consistent optical path length) : " << fcopls << " (blue dot)" << std::endl;

	lens.draw(board, LibBoard::Color::DarkCyan);

	auto ray_traces = trace_rays_through_lens(lens, 1., refr_ind, 9, RAYS_DISTANCE, 2);

	std::array<LibBoard::Color, 2> colors = {
		LibBoard::Color::Red,
		LibBoard::Color::Green
	};

	for (const auto& ray_trace : ray_traces)
	{
		for (size_t i = 0; i != ray_trace.size() - 1; ++i)
		{
			LibBoard::Color color = colors[i % colors.size()];
			vec_t end = ray_trace[i + 1].origin();
			ray_trace[i].draw(board, color, end);
		}
		ray_trace.back().draw(board, LibBoard::Color::Blue, 15);
	}

	board.setLineWidth(0.1);

	board.setPenColor(LibBoard::Color::Red);
	board.drawDot(fps, 0);

	board.setPenColor(LibBoard::Color::Green);
	board.drawDot(fpopls, 0);

	board.setPenColor(LibBoard::Color::Blue);
	board.drawDot(fcopls, 0);

	board.saveSVG("focus.svg");
	std::cout << "Lens illustration saved as focus.svg" << std::endl;

	return 0;
}