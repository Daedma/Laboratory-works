#include "View.hpp"

int main(int argc, char const* argv[])
{
	View view{};
	while (!view.isDone())
	{
		view.draw();
	}
	return 0;
}
