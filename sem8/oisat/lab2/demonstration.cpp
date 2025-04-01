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
	std::cout << "Enter the refractive index 1 (before refraction): ";
	std::cin >> refr_ind_out;
	std::cout << "Enter the refractive index 2 (after refraction): ";
	std::cin >> refr_ind_in;

	try
	{
		ray r(origin, direction);
		LibBoard::Board board;
		board.setFillColor(LibBoard::Color::White);
		board.setPenColor(LibBoard::Color::White);
		board.setLineWidth(10.);
		board.fillRectangle(-1000, -1000, 2000, 2000);
		board.moveCenter({ 0., 0. });
		surface->draw(board, LibBoard::Color::Black);

		board.setLineWidth(1.);

		const int max_iterations = 10;

		auto reflected_ray = surface->reflect_ray(r);
		if(reflected_ray)
		{
			reflected_ray->draw(board, LibBoard::Color::Red, 
				reflected_ray->origin() + reflected_ray->direction() * 500.);
		}

		for (int i = 0; i != max_iterations; ++i)
		{
			auto refracted_ray = surface->refract_ray(r, refr_ind_out, refr_ind_in);

			if (refracted_ray)
			{
				vec_t intersection_point = refracted_ray->origin();
				r.draw(board, LibBoard::Color::Blue, intersection_point);

				if (glm::dot(r.direction(), refracted_ray->direction()) > 0.)
				{
					std::swap(refr_ind_in, refr_ind_out);
				}
				r = *refracted_ray;
			}
			else
			{
				r.draw(board, LibBoard::Color::Blue, r.origin() + r.direction() * 500.);
				break;
			}
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