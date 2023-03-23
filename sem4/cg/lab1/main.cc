#include "SFML/Graphics.hpp"
#include <string>

void dragon(size_t count, sf::VertexArray& lines)
{
	if (count == 0) return;
	sf::Transform trans;
	trans.rotate(90, lines[lines.getVertexCount() - 1].position);
	for (int i = lines.getVertexCount() - 2; i != -1; --i)
	{
		lines.append({ { trans.transformPoint(lines[i].position) }, sf::Color::Cyan });
	}
	dragon(--count, lines);
}

int main(int argc, char const* argv[])
{
	size_t count = 10;
	if (argc == 2)
	{
		try
		{
			count = std::stoull(argv[1]);
		}
		catch (...) {}
	}
	sf::VertexArray lines{ sf::PrimitiveType::LineStrip };
	lines.append({ { 0, 0 }, sf::Color::Cyan });
	lines.append({ { 0, 1 }, sf::Color::Cyan });
	dragon(count, lines);
	sf::RenderWindow window(sf::VideoMode(800, 800), "Dragon");
	sf::FloatRect bounds = lines.getBounds();
	sf::Transform trans;
	float scale = std::min((window.getSize().x) / (bounds.width), (window.getSize().y) / (bounds.height));
	trans.scale(scale, scale);
	trans.translate(-bounds.left, -bounds.top);
	sf::RenderStates state(trans);
	window.draw(lines, state);
	window.display();
	while (window.isOpen())
	{

		// Process events
		sf::Event event;
		while (window.pollEvent(event))
		{
			// Close window: exit
			if (event.type == sf::Event::Closed)
				window.close();
		}
	}
	return 0;
}
