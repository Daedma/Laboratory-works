#include <iostream>
#include <memory>
#include <string>
#include <vector>
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

	biconvex_lens lens(100, 400, 50, 100, 200, 50);

	lens.draw(board, LibBoard::Color::Black);

	board.saveSVG("lens.svg");
	std::cout << "Lens illustration saved as lens.svg" << std::endl;

	return 0;
}