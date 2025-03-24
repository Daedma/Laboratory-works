#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <glm/trigonometric.hpp>

#include "shape.hpp"
#include "sphere.hpp"
#include "plane.hpp"
#include "ellipse.hpp"

namespace
{
	std::unique_ptr<shape> create_shape(const std::string& shape_type)
	{
		if (shape_type == "sphere")
		{
			double radius;
			std::cout << "Enter the radius of the sphere: ";
			std::cin >> radius;
			return std::make_unique<sphere>(radius);
		}
		else if (shape_type == "plane")
		{
			auto plane_ = std::make_unique<plane>();
			plane_->rotation({glm::radians(90.), 0. ,0.});
			return std::make_unique<plane>();
		}
		else if (shape_type == "ellipse")
		{
			double a, b, c;
			std::cout << "Enter a, b, c: ";
			std::cin >> a >> b >> c;
			return std::make_unique<ellipse>(a, b, c);
		}
		return nullptr;
	}

	void get_ray_input(vec_t& origin, vec_t& direction)
	{
		std::cout << "Enter the ray origin (x y z): ";
		std::cin >> origin.x >> origin.y >> origin.z;
		std::cout << "Enter the ray direction (dx dy dz): ";
		std::cin >> direction.x >> direction.y >> direction.z;
	}
}