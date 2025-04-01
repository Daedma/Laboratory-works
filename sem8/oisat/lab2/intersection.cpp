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
        auto intersection_points = surface->intersection_points(r);

        if (intersection_points.empty())
        {
            std::cout << "No intersection points found." << std::endl;
        }
        else
        {
            std::cout << "Intersection points:" << std::endl;
            for (const auto& point : intersection_points)
            {
                std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}