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

	double refr_ind_out, refr_ind_in;
	std::cout << "Enter the refractive index outside the surface: ";
	std::cin >> refr_ind_out;
	std::cout << "Enter the refractive index inside the surface: ";
	std::cin >> refr_ind_in;

	try
	{
		ray r(origin, direction);
		auto refracted_ray = surface->refract_ray(r, refr_ind_out, refr_ind_in);

		if (!refracted_ray)
		{
			std::cout << "No refracted rays found." << std::endl;
		}
		else
		{
			std::cout << "Refracted ray:" << std::endl;
			std::cout << "Origin: (" << refracted_ray->origin().x << ", " << refracted_ray->origin().y << ", " << refracted_ray->origin().z << ")" << std::endl;
			std::cout << "Direction: (" << refracted_ray->direction().x << ", " << refracted_ray->direction().y << ", " << refracted_ray->direction().z << ")" << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}