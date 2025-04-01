#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ray.hpp"
#include "shape.hpp"
#include "utils.hpp"

int main()
{
	std::string shape_type;
	std::cout << "Enter the shape type (sphere/plane/ellipse): ";
	std::cin >> shape_type;

	auto surface = create_shape(shape_type);
	if (!surface)
	{
		std::cerr << "Invalid shape type!" << std::endl;
		return 1;
	}

	vec_t origin, direction;
	get_ray_input(origin, direction);

	try
	{
		ray r(origin, direction);
		auto reflected_ray = surface->reflect_ray(r);

		if (!reflected_ray)
		{
			std::cout << "No reflected rays found." << std::endl;
		}
		else
		{
			std::cout << "Reflected ray:" << std::endl;
			std::cout << "Origin: (" << reflected_ray->origin().x << ", " << reflected_ray->origin().y << ", " << reflected_ray->origin().z << ")" << std::endl;
			std::cout << "Direction: (" << reflected_ray->direction().x << ", " << reflected_ray->direction().y << ", " << reflected_ray->direction().z << ")" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}