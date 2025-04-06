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

	constexpr double PRECISION = 1.e-2;

	constexpr double RAYS_DISTANCE = 1.;

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
			if (!intersection.empty())
			{
				intersections.emplace_back(intersection.front());
			}
		}
		if(intersections.empty())
		{
			return std::numeric_limits<double>::max();
		}
		return sd(intersections, { 0., 0., z_ });
		};


	double min_z = lens_.minmax_z().second;
	double max_z = lens_.minmax_z().second + max_z_;

	return minimum(intersection_points_sd, min_z, max_z, PRECISION);
}

int main()
{
	LibBoard::Board board;
	board.setFillColor(LibBoard::Color::White);
	board.setPenColor(LibBoard::Color::White);
	board.setLineWidth(0.02);
	board.moveCenter({ 0., 0. });

	double target_focus;
	double refr_ind;

	std::cout << "Enter target focal distance: ";
	std::cin >> target_focus;
	std::cout << "Enter refractive index of lens: ";
	std::cin >> refr_ind;

	// Диапазоны параметров для перебора
	double a1_min = 1.0, a1_max = 4.0, a1_step = 1.0;
	double b1_min = 1.0, b1_max = 4.0, b1_step = 1.0;
	double a2_min = 1.0, a2_max = 4.0, a2_step = 1.0;
	double b2_min = 1.0, b2_max = 4.0, b2_step = 1.0;
	double center_1_min = 0.0, center_1_max = 1.0, center_1_step = 0.1;
	double center_2_min = 0.0, center_2_max = 1.0, center_2_step = 0.1;

	double best_a1 = 0, best_b1 = 0, best_a2 = 0, best_b2 = 0, best_center_1 = 0, best_center_2 = 0;
	double min_error = std::numeric_limits<double>::max();

	// Перебор параметров
	for (double a1 = a1_min; a1 <= a1_max; a1 += a1_step)
	{
		for (double b1 = b1_min; b1 <= b1_max; b1 += b1_step)
		{
			for (double a2 = a2_min; a2 <= a2_max; a2 += a2_step)
			{
				for (double b2 = b2_min; b2 <= b2_max; b2 += b2_step)
				{
					for (double center_1 = center_1_min; center_1 <= center_1_max; center_1 += center_1_step)
					{
						for (double center_2 = center_2_min; center_2 <= center_2_max; center_2 += center_2_step)
						{
							biconvex_lens lens(a1, b1, center_1, a2, b2, center_2);
							double calculated_focus = focus_ps(lens, refr_ind, target_focus);

							double error = std::abs(calculated_focus - target_focus);
							if (error < min_error)
							{
								min_error = error;
								best_a1 = a1;
								best_b1 = b1;
								best_a2 = a2;
								best_b2 = b2;
								best_center_1 = center_1;
								best_center_2 = center_2;
							}
						}
					}
				}
			}
		}
	}

	std::cout << "Optimal parameters found:" << std::endl;
	std::cout << "a1: " << best_a1 << ", b1: " << best_b1 << std::endl;
	std::cout << "a2: " << best_a2 << ", b2: " << best_b2 << std::endl;
	std::cout << "center_1: " << best_center_1 << ", center_2: " << best_center_2 << std::endl;
	std::cout << "Minimum error: " << min_error << std::endl;

	// Построение линзы с оптимальными параметрами
	biconvex_lens optimal_lens(best_a1, best_b1, best_center_1, best_a2, best_b2, best_center_2);
	optimal_lens.draw(board, LibBoard::Color::DarkCyan);

	board.saveSVG("optimal_lens.svg");
	std::cout << "Optimal lens illustration saved as optimal_lens.svg" << std::endl;

	return 0;
}