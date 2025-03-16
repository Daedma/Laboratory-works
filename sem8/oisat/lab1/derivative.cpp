#include "common.hpp"

inline const auto field = fields::sinamp(0.5, 0.5, 2.);

inline const auto filter = filters::derivative(0);

int main()
{
	app::demostrateFiltration(field, filter);
	return 0;
}