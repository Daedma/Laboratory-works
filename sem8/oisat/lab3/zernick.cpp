#include "common.hpp"

inline const auto field = fields::bacteries(1., 2., 2.5, 2.5, 1., 0.001);

inline const auto filter = filters::zernick(0.05);

int main()
{
	app::demostrateFiltration(field, filter);
	return 0;
}