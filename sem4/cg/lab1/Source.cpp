#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <fstream>
#include <limits>
#include <iostream>

void task1()
{
	size_t H = 1000, W = 1000;
	sf::Image result;
	result.create(H, W, sf::Color::Black);
	result.saveToFile("image1.png");

	result.create(H, W, sf::Color::White);
	result.saveToFile("image2.png");

	result.create(H, W, sf::Color{ 255, 0, 0, 255 });
	result.saveToFile("image3.png");

	result.create(H, W);
	for (size_t i = 0; i != H; ++i)
		for (size_t j = 0; j != W; ++j)
		{
			sf::Uint8 r = 255 * cos(i * j * 1.e-2);
			sf::Uint8 g = 255 * sin(j * i * 1.e-2);
			sf::Uint8 b = 255 * sin(i * j * 1.e-2) * cos(i * j * 1.e-2);
			result.setPixel(i, j, sf::Color{r, g, b, 255});
		}
	result.saveToFile("image4.png");
}



void line1(int x0, int y0, int x1, int y1,
	sf::Image& image, sf::Color color)
{
	for (float t = 0.0; t < 1.0; t += 0.01)
	{
		int x = x0 * (1. - t) + x1 * t;
		int y = y0 * (1. - t) + y1 * t;
		image.setPixel(x, y, color);
	}
}

void line2(int x0, int y0, int x1, int y1,
	sf::Image& image, sf::Color color)
{
	for (int x = x0; x <= x1; x++)
	{
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		image.setPixel(x, y, color);
	}
}

void line3(int x0, int y0, int x1, int y1, sf::Image& image, sf::Color
	color)
{
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1))
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1)
	{ // make it left-to-right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}
	for (int x = x0; x <= x1; x++)
	{
		float t = (x - x0) / (float)(x1 - x0);
		int y = y0 * (1. - t) + y1 * t;
		if (steep)
		{
			image.setPixel(y, x, color);
		}
		else
		{
			image.setPixel(x, y, color);
		}
	}
}

void line(int x0, int y0, int x1, int y1, sf::Image& image, sf::Color color)
{
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1))
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}
	if (x0 > x1)
	{ // make it left-to-right
		std::swap(x0, x1);
		std::swap(y0, y1);
	}
	int dx = x1 - x0;
	int dy = y1 - y0;
	float derror = std::abs(dy / float(dx));
	float error = 0;
	int y = y0;
	for (int x = x0; x <= x1; x++)
	{
		if (steep)
		{
			image.setPixel(y, x, color);
		}
		else
		{
			image.setPixel(x, y, color);
		}
		error += derror;
		if (error > .5)
		{
			y += (y1 > y0 ? 1 : -1);
			error -= 1.;
		}
	}
}

void task2()
{
	size_t H = 200, W = 200;
	sf::Image result1;
	sf::Image result2;
	sf::Image result3;
	sf::Image result;
	result1.create(W, H);
	result2.create(W, H);
	result3.create(W, H);
	result.create(W, H);
	for (size_t i = 0; i != 13; ++i)
	{
		float a = 2 * 3.14 * i / 13.;
		line1(100, 100, 100 + 95 * cos(a), 100 + 95 * sin(a), result1, sf::Color::Cyan);
		line2(100, 100, 100 + 95 * cos(a), 100 + 95 * sin(a), result2, sf::Color::Cyan);
		line3(100, 100, 100 + 95 * cos(a), 100 + 95 * sin(a), result3, sf::Color::Cyan);
		line(100, 100, 100 + 95 * cos(a), 100 + 95 * sin(a), result, sf::Color::Cyan);
	}
	result1.saveToFile("image5.png");
	result2.saveToFile("image6.png");
	result3.saveToFile("image7.png");
	result.saveToFile("image8.png");
}

struct Vertex
{
	float x, y, z;
};

std::istream& operator>>(std::istream& is, Vertex& rhs)
{
	return is >> rhs.x >> rhs.y >> rhs.z;
}

std::ostream& operator<<(std::ostream& os, const Vertex& rhs)
{
	return os << rhs.x << ' ' << rhs.y << ' ' << rhs.z;
}

struct Polygon
{
	Vertex v1, v2, v3;

	std::vector<Vertex>* v_list;
};

std::istream& operator>>(std::istream& is, Polygon& rhs)
{
	size_t n;
	is >> n;
	rhs.v1 = (*rhs.v_list)[n - 1];
	is.clear();
	is.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
	is >> n;
	rhs.v2 = (*rhs.v_list)[n - 1];
	is.clear();
	is.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
	is >> n;
	rhs.v3 = (*rhs.v_list)[n - 1];
	is.clear();
	is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	return is;
}

class Object3d
{
	std::vector<Vertex> m_vertexes;
	std::vector<Polygon> m_polygons;

	sf::Vector3f m_min_corner;
	sf::Vector3f m_max_corner;
public:
	enum class Print
	{
		VERTEX_ONLY,
		LINES_ONLY
	};

	void load(const std::string& filename)
	{
		std::ifstream ifs(filename);
		std::string item;
		Vertex cur;
		Polygon poly;
		poly.v_list = &m_vertexes;
		while (ifs >> item)
		{
			if (item == "#")
			{
				ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
			else if (item == "v")
			{
				ifs >> cur;
				m_vertexes.emplace_back(cur);
			}
			else if (item == "f")
			{
				ifs >> poly;
				m_polygons.emplace_back(poly);
			}
		}
		std::cout << m_polygons.size() << '\n';
		calc_aabb();
	}

	void print(const std::string& filename, Print type)
	{
		switch (type)
		{
		case Print::VERTEX_ONLY:
			print_vertex(filename);
			break;
		case Print::LINES_ONLY:
			print_lines(filename);
			break;
		}
	}

private:
	void print_lines(const std::string& filename)
	{
		const size_t height = 1000, width = 1000;
		sf::Image result;
		result.create(width, height);
		float scale_x = width / std::abs(m_max_corner.x - m_min_corner.x) * 0.75f;
		float scale_y = height / std::abs(m_max_corner.y - m_min_corner.y) * 0.75f;
		float scale = std::min(scale_x, scale_y);
		float shift_x = -m_min_corner.x * scale + width / 8;
		float shift_y = -m_min_corner.y * scale + height / 8;
		auto trans_x = [scale, shift_x, width](float x) {
			return width - (x * scale + shift_x);
		};
		auto trans_y = [scale, shift_y, height](float y) {
			return height - (y * scale + shift_y);
		};
		sf::Color color = sf::Color::Yellow;
		for (const auto& i : m_polygons)
		{
			line(trans_x(i.v1.x), trans_y(i.v1.y), trans_x(i.v2.x), trans_y(i.v2.y), result, color);
			line(trans_x(i.v2.x), trans_y(i.v2.y), trans_x(i.v3.x), trans_y(i.v3.y), result, color);
			line(trans_x(i.v3.x), trans_y(i.v3.y), trans_x(i.v1.x), trans_y(i.v1.y), result, color);
		}
		result.saveToFile(filename);
	}

	void print_vertex(const std::string& filename)
	{
		const size_t height = 1000, width = 1000;
		sf::Image result;
		result.create(width, height);
		float scale_x = width / std::abs(m_max_corner.x - m_min_corner.x) * 0.75f;
		float scale_y = height / std::abs(m_max_corner.y - m_min_corner.y) * 0.75f;
		float scale = std::min(scale_x, scale_y);
		float shift_x = -m_min_corner.x * scale + width / 8;
		float shift_y = -m_min_corner.y * scale + height / 8;
		for (const auto& i : m_vertexes)
		{
			result.setPixel(width - i.x * scale - shift_x, height - i.y * scale - shift_y, sf::Color::Yellow);
		}
		result.saveToFile(filename);
	}

	void calc_aabb()
	{
		m_max_corner = { -INFINITY, -INFINITY, -INFINITY };
		m_min_corner = { INFINITY, INFINITY, INFINITY };
		for (const auto& i : m_vertexes)
		{
			if (m_max_corner.x < i.x)
				m_max_corner.x = i.x;
			if (m_max_corner.y < i.y)
				m_max_corner.y = i.y;
			if (m_max_corner.z < i.z)
				m_max_corner.z = i.z;

			if (m_min_corner.x > i.x)
				m_min_corner.x = i.x;
			if (m_min_corner.y > i.y)
				m_min_corner.y = i.y;
			if (m_min_corner.z > i.z)
				m_min_corner.z = i.z;
		}
	}

	void line(int x0, int y0, int x1, int y1, sf::Image& image, sf::Color color)
	{
		bool steep = false;
		if (std::abs(x0 - x1) < std::abs(y0 - y1))
		{
			std::swap(x0, y0);
			std::swap(x1, y1);
			steep = true;
		}
		if (x0 > x1)
		{ // make it left-to-right
			std::swap(x0, x1);
			std::swap(y0, y1);
		}
		int dx = x1 - x0;
		int dy = y1 - y0;
		float derror = std::abs(dy / float(dx));
		float error = 0;
		int y = y0;
		for (int x = x0; x <= x1; x++)
		{
			if (steep)
			{
				image.setPixel(y, x, color);
			}
			else
			{
				image.setPixel(x, y, color);
			}
			error += derror;
			if (error > .5)
			{
				y += (y1 > y0 ? 1 : -1);
				error -= 1.;
			}
		}
	}

};

int main()
{
	task1();
	task2();
	Object3d obj;
	obj.load("model_2.obj");
	obj.print("image9.png", Object3d::Print::VERTEX_ONLY);
	obj.print("image10.png", Object3d::Print::LINES_ONLY);
}