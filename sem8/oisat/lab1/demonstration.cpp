#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <Board.h>

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

		LibBoard::Board board;
		board.setFillColor("white");
		board.fillRectangle(-1000, -1000, 2000, 2000);
		surface->draw(board, LibBoard::Color::Black); // Draw the shape

		if (refracted_ray)
		{
			// Draw the incident ray up to the intersection point
			vec_t intersection_point = refracted_ray->origin();
			r.draw(board, LibBoard::Color::Blue, intersection_point);

			// Draw the refracted ray from the intersection point
			refracted_ray->draw(board, LibBoard::Color::Red, intersection_point + refracted_ray->direction() * 100.0f); // Extend the refracted ray for visualization
		}
		else
		{
			// If no intersection, draw the incident ray fully
			r.draw(board, LibBoard::Color::Blue, r.origin() + r.direction() * 1000.0f); // Extend the incident ray for visualization
		}

		board.saveSVG("refraction.svg");
		std::cout << "Refraction illustration saved as refraction.svg" << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}