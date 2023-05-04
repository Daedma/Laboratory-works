#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <Mathter/Matrix.hpp>
#include <Mathter/Vector.hpp>
#include <Mathter/Quaternion.hpp>
#include <cmath>
#include <vector>
#include <memory>
#include <fstream>
#include <limits>
#include <random>
#include <iostream>
#include <iterator>

using Vector3f = mathter::Vector<double, 3>;

using Quat = mathter::Quaternion<double>;

using Vector3ld = mathter::Vector<long double, 3>;

using Vector2f = mathter::Vector<double, 2>;

using Mat44 = mathter::Matrix<double, 4, 4, mathter::eMatrixOrder::PRECEDE_VECTOR>;

using Mat33 = mathter::Matrix<double, 3, 3, mathter::eMatrixOrder::PRECEDE_VECTOR>;


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

Vector3f rotate_point(const Vector3f& point, const Quat& rot)
{
	Quat p(point);
	Quat res = rot * p * mathter::Conjugate(rot);
	return res.VectorPart();
}

double dot_product(const sf::Vector3f& lhs, const sf::Vector3f& rhs)
{
	return rhs.x * lhs.x + rhs.y * lhs.y + rhs.z * lhs.z;
}

Vector3ld to_barycentric(int x, int y, long double x0, long double y0, long  double x1, long double y1, long double x2, long double y2)
{
	long double lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2));
	long double lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0));
	long double lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1));
	return { lambda0, lambda1, lambda2 };
}


struct Vertex
{
	double x, y, z;

	Vertex() = default;
	Vertex(const Vector2f& vec): x(vec.x), y(vec.y), z(1.f) {}
	Vertex(const Vector3f& vec): x(vec.x), y(vec.y), z(vec.z) {}
	Vertex(const sf::Vector2f& vec): x(vec.x), y(vec.y), z(1.f) {}
	Vertex(const sf::Vector3f& vec): x(vec.x), y(vec.y), z(vec.z) {}

	operator sf::Vector3f() const
	{
		return { x, y, z };
	}

	operator Vector3f() const
	{
		return { x, y, z };
	}

	bool operator!=(const Vertex& other) const
	{
		return x != other.x && y != other.y && z != other.z;
	}

	bool operator==(const Vertex& other) const
	{
		return !(*this != other);
	}
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

	Vector3f vn0, vn1, vn2;

	Vector3f norm;

	Vector3f get_normal() const
	{
		double x = (v1.y - v0.y) * (v1.z - v2.z) - (v1.z - v0.z) * (v1.y - v2.y);
		double y = (v1.z - v0.z) * (v1.x - v2.x) - (v1.x - v0.x) * (v1.z - v2.z);
		double z = (v1.x - v0.x) * (v1.y - v2.y) - (v1.y - v0.y) * (v1.x - v2.x);
		Vector3f result{ x, y, z };
		result *= Q_rsqrt(x * x + y * y + z * z);
		return result;
	}

	Polygon() = default;

	Polygon(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2):
		v0(_v0), v1(_v1), v2(_v2), norm(get_normal())
	{}

	Polygon(const Vertex& _v0, const Vertex& _v1, const Vertex& _v2, const Vector3f& _vn0, const Vector3f& _vn1, const Vector3f& _vn2):
		v0(_v0), v1(_v1), v2(_v2), vn0(_vn0), vn1(_vn1), vn2(_vn2), norm(get_normal())
	{}

	bool operator!=(const Polygon& other) const
	{
		return  v0 != other.v0 && v1 != other.v1 && v2 != other.v2;
	}
};

class Object3d
{
	std::vector<Polygon> m_polygons;

	std::vector<Vector3f> m_normals;

	Vector3f m_light = { 0.f, 0.f, 1.f };

	Vector3f m_min_corner;
	Vector3f m_max_corner;
public:
	using array2d = std::vector<std::vector<double>>;

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
				if (!m_normals.empty())
				{
					size_t v0, v1, v2;
					size_t vn0, vn1, vn2;

					ifs >> v0;
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs >> vn0;

					ifs >> v1;
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs >> vn1;

					ifs >> v2;
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs.ignore(std::numeric_limits<std::streamsize>::max(), '/');
					ifs >> vn2;
					m_polygons.emplace_back(vertexes[v0 - 1], vertexes[v1 - 1], vertexes[v2 - 1],
						m_normals[vn0 - 1], m_normals[vn1 - 1], m_normals[vn2 - 1]);
				}
				else
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
			else if (item == "vn")
			{
				double	nx, ny, nz;
				ifs >> nx >> ny >> nz;
				m_normals.emplace_back(nx, ny, nz);
			}
		}
		calc_aabb(vertexes);
	}

	void print(sf::Image& dest, const Mat33& scale = mathter::Identity(), const Vector3f& translate = (Vector3f(0.)), const Quat& rotation = mathter::Identity())
	{
		print_polygons(dest, scale, translate, rotation);
	}

	Vector3f getDefaultTranslate() const { return { 0., 0., -m_min_corner.z }; }

private:
	void print_polygons(sf::Image& dest, const Mat33& scale = mathter::Identity(), const Vector3f& translate = (Vector3f(0.)), const Quat& rotation = mathter::Identity())
	{
		array2d z_buffer(dest.getSize().y, (std::vector<double>(dest.getSize().x, INFINITY)));
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<uint32_t> d(0x0, 0xFF);
		sf::Color color(d(gen), d(gen), d(gen));
		std::vector<Polygon> rotates;
		rotates.resize(m_polygons.size());
		std::transform(m_polygons.begin(), m_polygons.end(), rotates.begin(),
			[rotation](const Polygon& val) {
				return Polygon{
					rotate_point(val.v0, rotation),
					rotate_point(val.v1, rotation),
					rotate_point(val.v2, rotation),
					rotate_point(val.vn0, rotation),
					rotate_point(val.vn1, rotation),
					rotate_point(val.vn2, rotation)
				};
			});
		for (const auto& i : rotates)
		{
			print_polygon(i, dest, color, scale, translate, z_buffer);
		}
	}

	void print_polygon(const Polygon& poly, sf::Image& image, const sf::Color& color, const Mat33& scale, const Vector3f& translate, array2d& z_buffer)
	{
		static constexpr double eps = -0.2;
		double poly_light_cos = mathter::Dot(poly.get_normal(), m_light);
		double l1 = mathter::Dot(poly.vn0, m_light);
		double l2 = mathter::Dot(poly.vn1, m_light);
		double l3 = mathter::Dot(poly.vn2, m_light);
		if (poly_light_cos >= 0) return;
		Polygon scaled_poly{ scale * Vector3f{ poly.v0.x, poly.v0.y, 1 },
			scale * Vector3f{ poly.v1.x, poly.v1.y, 1 },
			scale * Vector3f{ poly.v2.x, poly.v2.y, 1 } };

		sf::Color outcolor(color.r * poly_light_cos, color.g * poly_light_cos, color.b * poly_light_cos);

		sf::Vector2i max_corner{
			std::max({ scaled_poly.v0.x, scaled_poly.v1.x, scaled_poly.v2.x }),
				std::max({ scaled_poly.v0.y, scaled_poly.v1.y, scaled_poly.v2.y })};
		sf::Vector2i min_corner{
			std::min({ scaled_poly.v0.x, scaled_poly.v1.x, scaled_poly.v2.x }),
				std::min({ scaled_poly.v0.y, scaled_poly.v1.y, scaled_poly.v2.y })};
		max_corner.x = std::clamp<int>(max_corner.x, 0, image.getSize().x);
		max_corner.y = std::clamp<int>(max_corner.y, 0, image.getSize().y);
		min_corner.x = std::clamp<int>(min_corner.x, 0, image.getSize().x);
		min_corner.y = std::clamp<int>(min_corner.y, 0, image.getSize().y);

		for (int i = min_corner.x; i != max_corner.x; ++i)
			for (int j = min_corner.y; j != max_corner.y; ++j)
			{
				Vector3ld bar_coord = to_barycentric(i, j,
					scaled_poly.v0.x, scaled_poly.v0.y,
					scaled_poly.v1.x, scaled_poly.v1.y,
					scaled_poly.v2.x, scaled_poly.v2.y);
				if (bar_coord.x >= eps && bar_coord.y >= eps && bar_coord.z >= eps)
				{
					long double z = bar_coord.x * poly.v0.z + bar_coord.y * poly.v1.z + bar_coord.z * poly.v2.z;
					if (-z < z_buffer[j][i])
					{
						double out = (bar_coord.x * l1 + bar_coord.y * l2 + bar_coord.z * l3);
						image.setPixel(i, j, sf::Color{color.r* out, color.g* out, color.b* out});
						z_buffer[j][i] = -z;
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

	void calc_normals()
	{}
};

int main()
{
	sf::Image image;
	image.create(1000, 1000);
	Object3d model_1;

	Mat33 transform(
		-50000, 0, 500,
		0, -50000, 1500,
		0, 0, 1
	);
	model_1.load("model_1.obj");

	image.create(1000, 1000);
	model_1.print(image, transform, model_1.getDefaultTranslate());
	// model_1.print(image, transform, { 0.005, -0.045, 15.0 });
	image.saveToFile("model_1.png");

	Mat33 transform3(
		-10000, 0, 500,
		0, -10000, 500,
		0, 0, 1
	);

	image.create(1000, 1000);
	model_1.print(image, transform3, model_1.getDefaultTranslate());
	// model_1.print(image, transform, { 0.005, -0.045, 15.0 });
	image.saveToFile("model_1_origin.png");

	image.create(1000, 1000);
	Mat33 transform2(
		-10000, 0, 500,
		0, -10000, 500,
		0, 0, 1
	);
	model_1.print(image, transform2, model_1.getDefaultTranslate(),
		mathter::RotationEuler(0.l, mathter::Deg2Rad(90), 0.l));
	image.saveToFile("model_1(rotated).png");
}