#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <fstream>
#include <limits>
#include <random>
#include <iostream>

sf::Vector3f to_barycentric(int x, int y, float x0, float y0, float x1, float y1, float x2, float y2)
{
	float lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2));
	float lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0));
	float lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1));
	return { lambda0, lambda1, lambda2 };
}

struct Vertex
{
	float x, y, z;
	Vertex() = default;
	Vertex(const sf::Vector2f& vec): x(vec.x), y(vec.y), z(0.f) {}
	Vertex(const sf::Vector3f& vec): x(vec.x), y(vec.y), z(vec.z) {}
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
	Vertex v0, v1, v2;

	Polygon() = default;

	Polygon(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2):
		v0(_v0), v1(_v1), v2(_v2)
	{}
};


class Object3d
{
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
		m_polygons.clear();
		std::ifstream ifs(filename);
		std::string item;
		std::vector<Vertex> vertexes;
		Vertex cur;
		Polygon poly;
		while (ifs >> item)
		{
			if (item == "#")
			{
				ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
			else if (item == "v")
			{
				ifs >> cur;
				vertexes.emplace_back(cur);
			}
			else if (item == "f")
			{
				size_t v0, v1, v2;
				ifs >> v0;
				ifs.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
				ifs >> v1;
				ifs.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
				ifs >> v2;
				ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				m_polygons.emplace_back(vertexes[v0 - 1], vertexes[v1 - 1], vertexes[v2 - 1]);
			}
		}
		calc_aabb(vertexes);
	}

	void print(sf::Image& dest)
	{
		print_polygons(dest);
	}

private:
	void print_polygons(sf::Image& dest)
	{
		sf::Transform trans;
		float scale_x = dest.getSize().x / std::abs(m_max_corner.x - m_min_corner.x) * 0.75f;
		float scale_y = dest.getSize().y / std::abs(m_max_corner.y - m_min_corner.y) * 0.75f;
		float scale = std::min(scale_x, scale_y);
		trans.scale(scale, scale);
		trans.translate(-m_min_corner.x, -m_min_corner.y);
		// trans.translate(dest.getSize().x / 8, dest.getSize().y / 8);
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<uint32_t> d(0xFF'00'00, 0xFF'FF'FF);
		for (const auto& i : m_polygons)
		{
			sf::Color color(d(gen) * 8 + 0xFF);
			Polygon poly{ trans.transformPoint(i.v0.x, i.v0.y),
				trans.transformPoint(i.v1.x, i.v1.y),
				trans.transformPoint(i.v2.x, i.v2.y) };
			print_polygon(poly, dest, color);
		}
	}

	void print_polygon(const Polygon& poly, sf::Image& image, const sf::Color& color)
	{
		// sf::Vector2i max_corner{ static_cast<int>(std::max({ 0.f, poly.v0.x, poly.v1.x, poly.v2.x })),
			// static_cast<int>(std::max({ 0.f, poly.v0.y, poly.v1.y, poly.v2.y })) };
		// sf::Vector2i min_corner{static_cast<int>(std::min({ static_cast<float>(image.getSize().x), poly.v0.x, poly.v1.x, poly.v2.x })),
			// static_cast<int>(std::min({ static_cast<float>(image.getSize().y), poly.v0.y, poly.v1.y, poly.v2.y }))};
		sf::Vector2i max_corner{ static_cast<int>(std::max({ poly.v0.x, poly.v1.x, poly.v2.x })),
			static_cast<int>(std::max({ poly.v0.y, poly.v1.y, poly.v2.y })) };
		sf::Vector2i min_corner{static_cast<int>(std::min({ poly.v0.x, poly.v1.x, poly.v2.x })),
			static_cast<int>(std::min({ poly.v0.y, poly.v1.y, poly.v2.y }))};
		if (min_corner.x > image.getSize().x || min_corner.y > image.getSize().y || max_corner.x < 0 || max_corner.y < 0) return;
		if (min_corner.x < 0) min_corner.x = 0;
		if (min_corner.y < 0) min_corner.y = 0;
		if (max_corner.x > image.getSize().x) max_corner.x = image.getSize().x;
		if (max_corner.y > image.getSize().y) max_corner.y = image.getSize().y;
		for (int i = min_corner.x; i != max_corner.x; ++i)
			for (int j = min_corner.y; j != max_corner.y; ++j)
			{
				if (j > image.getSize().y || i > image.getSize().x || i < 0 || j < 0)
				{
					std::cout << i << ' ' << j << '\n';
				}
				sf::Vector3f bar_coord = to_barycentric(i, j,
					poly.v0.x, poly.v0.y,
					poly.v1.x, poly.v1.y,
					poly.v2.x, poly.v2.y);
				if (bar_coord.x > 0 && bar_coord.y > 0 && bar_coord.z > 0)
				{
					// image.setPixel(i, j, color);
				}
			}

	}

	void calc_aabb(const std::vector<Vertex>& vertexes)
	{
		m_max_corner = { -INFINITY, -INFINITY, -INFINITY };
		m_min_corner = { INFINITY, INFINITY, INFINITY };
		for (const auto& i : vertexes)
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
};

int main()
{
	sf::Image image;
	image.create(100, 100);
	// Object3d obj;
	// obj.load("model_2.obj");
	std::cout << "obj is loaded\n";
	// obj.print(image);
	std::cout << "obj is printed\n";
	std::cout << image.saveToFile("result.png") << '\n';
}