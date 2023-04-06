#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <Mathter/Matrix.hpp>
#include <Mathter/Vector.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <fstream>
#include <limits>
#include <random>
#include <iostream>

float Q_rsqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y = number;
	i = *(long*)&y;                       // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1);               // what the fuck? 
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

float dot_product(const sf::Vector3f& lhs, const sf::Vector3f& rhs)
{
	return rhs.x * lhs.x + rhs.y * lhs.y + rhs.z * lhs.z;
}

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

	sf::Vector3f get_normal() const
	{
		float x = (v1.y - v0.y) * (v1.z - v2.z) - (v1.z - v0.z) * (v1.y - v2.y);
		float y = (v1.z - v0.z) * (v1.x - v2.x) - (v1.x - v0.x) * (v1.z - v2.z);
		float z = (v1.x - v0.x) * (v1.y - v2.y) - (v1.y - v0.y) * (v1.x - v2.x);
		sf::Vector3f result{ x, y, z };
		result *= Q_rsqrt(x * x + y * y + z * z);
		return result;
	}

	Polygon() = default;

	Polygon(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2):
		v0(_v0), v1(_v1), v2(_v2)
	{}
};

class Object3d
{
	std::vector<Polygon> m_polygons;

	sf::Vector3f m_light = { 0.f, 0.f, 0.1f };

	sf::Vector3f m_min_corner;
	sf::Vector3f m_max_corner;
public:
	using array2d = std::vector<std::vector<float>>;

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
		array2d z_buffer(dest.getSize().y, (std::vector<float>(dest.getSize().x, INFINITY)));
		float scale_x = dest.getSize().x / std::abs(m_max_corner.x - m_min_corner.x) * 0.75f;
		float scale_y = dest.getSize().y / std::abs(m_max_corner.y - m_min_corner.y) * 0.75f;
		float scale = std::min(scale_x, scale_y);
		trans.translate(dest.getSize().x, dest.getSize().y);
		trans.scale(-1.f, -1.f);
		trans.translate(dest.getSize().x / 8, dest.getSize().y / 8);
		trans.scale(scale, scale);
		trans.translate(-m_min_corner.x, -m_min_corner.y);
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<uint32_t> d(0x0, 0xFF);
		sf::Color color(d(gen), d(gen), d(gen));
		for (const auto& i : m_polygons)
		{
			Polygon poly{ trans.transformPoint(i.v0.x, i.v0.y),
				trans.transformPoint(i.v1.x, i.v1.y),
				trans.transformPoint(i.v2.x, i.v2.y) };
			poly.v0.z = -i.v0.z * scale;
			poly.v1.z = -i.v1.z * scale;
			poly.v2.z = -i.v2.z * scale;
			print_polygon(poly, dest, color, i.get_normal(), z_buffer);
		}
	}

	void print_polygon(const Polygon& poly, sf::Image& image, const sf::Color& color, const sf::Vector3f& normal, array2d& z_buffer)
	{
		static constexpr float eps = 0;
		float poly_light_cos = dot_product(normal, m_light);
		if (poly_light_cos >= 0) return;
		sf::Color outcolor(-color.r * poly_light_cos, -color.g * poly_light_cos, -color.b * poly_light_cos);
		sf::Vector2i max_corner{ static_cast<int>(std::max({ 0.f, poly.v0.x, poly.v1.x, poly.v2.x })),
			static_cast<int>(std::max({ 0.f, poly.v0.y, poly.v1.y, poly.v2.y })) };
		sf::Vector2i min_corner{static_cast<int>(std::min({ static_cast<float>(image.getSize().x), poly.v0.x, poly.v1.x, poly.v2.x })),
			static_cast<int>(std::min({ static_cast<float>(image.getSize().y), poly.v0.y, poly.v1.y, poly.v2.y }))};
		// sf::Vector2i max_corner{ static_cast<int>(std::max({ poly.v0.x, poly.v1.x, poly.v2.x })),
		// 	static_cast<int>(std::max({ poly.v0.y, poly.v1.y, poly.v2.y })) };
		// sf::Vector2i min_corner{static_cast<int>(std::min({ poly.v0.x, poly.v1.x, poly.v2.x })),
		// 	static_cast<int>(std::min({ poly.v0.y, poly.v1.y, poly.v2.y }))};
		// if (min_corner.x > image.getSize().x || min_corner.y > image.getSize().y || max_corner.x < 0 || max_corner.y < 0) return;
		// if (min_corner.x < 0) min_corner.x = 0;
		// if (min_corner.y < 0) min_corner.y = 0;
		// if (max_corner.x > image.getSize().x) max_corner.x = image.getSize().x;
		// if (max_corner.y > image.getSize().y) max_corner.y = image.getSize().y;
		for (int i = min_corner.x; i != max_corner.x; ++i)
			for (int j = min_corner.y; j != max_corner.y; ++j)
			{
				sf::Vector3f bar_coord = to_barycentric(i, j,
					poly.v0.x, poly.v0.y,
					poly.v1.x, poly.v1.y,
					poly.v2.x, poly.v2.y);
				if (bar_coord.x >= eps && bar_coord.y >= eps && bar_coord.z >= eps)
				{
					float z = bar_coord.x * poly.v0.z + bar_coord.y * poly.v1.z + bar_coord.z * poly.v2.z;
					if (z < z_buffer[j][i])
					{
						image.setPixel(i, j, outcolor);
						z_buffer[j][i] = z;
					}
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
	image.create(1000, 1000);
	Object3d model_1;
	model_1.load("model_1.obj");
	model_1.print(image);
	image.saveToFile("model_1.png");
	image.create(1000, 1000);
	model_1.load("model_2.obj");
	model_1.print(image);
	image.saveToFile("model_2.png");

}