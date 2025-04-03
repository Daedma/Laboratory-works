#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <array>

#include <Board.h>

#include "ray.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include "biconvex_lens.hpp"

int main()
{
	LibBoard::Board board;
	board.setFillColor(LibBoard::Color::White);
	board.setPenColor(LibBoard::Color::White);
	board.setLineWidth(2.);
	board.fillRectangle(-1000, -1000, 2000, 2000);
	board.moveCenter({ 0., 0. });

	biconvex_lens lens(80, 300, 50, 100, 300, 50);

	lens.draw(board, LibBoard::Color::Black);

	auto raytraces = trace_rays_through_lens(lens, 1., 1.5, 9, 300, 2);

	std::array<LibBoard::Color, 3> colors = {
		LibBoard::Color::Red,
		LibBoard::Color::Green,
		LibBoard::Color::Blue
	};

	for (const auto& raytrace : raytraces)
	{
		for (size_t i = 0; i != raytrace.size() - 1; ++i)
		{
			LibBoard::Color color = colors[i % colors.size()];
			vec_t end = raytrace[i + 1].origin();
			raytrace[i].draw(board, color, end);
		}
		raytrace.back().draw(board, LibBoard::Color::Cyan, 500);
	}

	board.saveSVG("lens.svg");
	std::cout << "Lens illustration saved as lens.svg" << std::endl;

	return 0;
}