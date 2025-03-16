#include "common.hpp"

inline const auto field = fields::gauss;

inline const auto filter = filters::identity;

int main()
{
	app::demostrateFiltration(field, filter);
	return 0;
}