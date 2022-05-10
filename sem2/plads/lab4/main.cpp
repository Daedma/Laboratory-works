/*
1. Определить поля и методы базового класса, указанного в задании.
2. Реализовать методы инициализации и удаления объекта
a. конструктор по умолчанию
b. конструктор копирования
c. конструктор инициализации объекта
d. деструктор
3. Реализовать методы вывода объекта в поток и чтения из потока в виде
перегруженных операций.
4. Определить операцию присваивания.
5. При необходимости переопределить операции выделения памяти для
возможности создания динамического массива объектов.
6. Реализовать методы, указанные в задании. При возможности использовать
перегрузку стандартных операций.
7. Определить операции сравнения: >, <, <=, >=, = =.
8. Подготовить тестовые примеры с проверкой реализованного функционала.

6. Прямоугольник

1) Изменение координаты
центра.
2) Изменение длины стороны.
3) Вычисление площади.
4) Вычисление периметра.
5) Сложение
прямоугольников.

Входные данные:
Координаты центра и длины сторон.
В методе 1: новая координата.
В методе 2: длина новой стороны.
Примечание:
Метод 5 реализуются через
перегрузку. Как сумма сторон и
средняя точка между центрами.
Результат выполнения:
Метод 1 – координаты центра.
Методы 2-5 – число.
*/
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>
#include <array>

class Rectangle
{
    double _width, _height;
    struct Point
    {
        double x, y;
    } _center;

public:
    Rectangle() noexcept : _width{0}, _height{0}, _center{0, 0} {}
    Rectangle(const Rectangle &rhs) noexcept : _width{rhs._width}, _height{rhs._height}, _center{rhs._center} {}
    Rectangle(double width, double height, Point center) : _width{width}, _height{height}, _center{center}
    {
        if (width < 0 || height < 0)
        {
            _width = _height = 0;
            throw std::invalid_argument{"rectangle sides cannot be negative"};
        }
    }
    ~Rectangle() noexcept {}

    Rectangle &operator=(const Rectangle &rhs) noexcept
    {
        _width = rhs._width;
        _height = rhs._height;
        _center = rhs._center;
        return *this;
    }
    bool operator<(const Rectangle &rhs) const noexcept { return area() < rhs.area(); }
    bool operator<=(const Rectangle &rhs) const noexcept { return area() <= rhs.area(); }
    bool operator>(const Rectangle &rhs) const noexcept { return area() > rhs.area(); }
    bool operator>=(const Rectangle &rhs) const noexcept { return area() >= rhs.area(); }
    bool operator==(const Rectangle &rhs) const noexcept
    {
        return _width == rhs._width && _height == rhs._height && _center.x == rhs._center.x && _center.y == rhs._center.y;
    }
    bool operator!=(const Rectangle &rhs) const noexcept { return !(*this == rhs); }

    double width() const noexcept { return _width; }
    double height() const noexcept { return _width; }

    double setw(double width)
    {
        if (width < 0)
            throw std::invalid_argument{"width cannot be negative"};
        return _width = width;
    }
    double seth(double height)
    {
        if (height < 0)
            throw std::invalid_argument{"height cannot be negative"};
        return _height = height;
    }
    void set_size(double width, double height)
    {
        *this = Rectangle{width, height, _center};
    }
    Point &set_coord(double x, double y) noexcept
    {
        _center.x = x;
        _center.y = y;
        return _center;
    }
    Point &center() noexcept { return _center; }
    const Point &center() const noexcept { return _center; }

    double area() const noexcept { return _width * _height; }
    double perimeter() const noexcept { return (_width + _height) * 2; }
    Rectangle operator+(const Rectangle &rhs) const noexcept
    {
        return {_width + rhs._width, _height + rhs._height, {(_center.x + rhs._center.x) / 2, (_center.y + rhs._center.y) / 2}};
    }
};

std::ostream &operator<<(std::ostream &os, const Rectangle &rhs)
{
    return os << rhs.width() << ' ' << rhs.height() << ' ' << rhs.center().x << ' ' << rhs.center().y;
}

std::istream &operator>>(std::istream &is, Rectangle &rhs)
{
    double width, height, x, y;
    if (is >> width >> height >> x >> y)
    {
        try
        {
            rhs.set_size(width, height);
            rhs.set_coord(x, y);
        }
        catch (...)
        {
            is.setstate(std::ios::failbit);
        }
    }
    return is;
}

int main()
{
    try
    {
        std::ifstream ifs{"input.txt"};
        ifs.exceptions(std::ios::failbit | std::ios::badbit);
        uint16_t mode;
        ifs >> mode;
        static const std::array<std::function<void(void)>, 6> modes{
            [&ifs]()
            {
                Rectangle rec1, rec2;
                ifs >> rec1 >> rec2;
                ifs.close();
                std::ofstream ofs{"output.txt"};
                ofs << (rec1 == rec2);
                ofs.close();
            },
            [&ifs]()
            {
                Rectangle rect;
                ifs >> rect;
                double nx, ny;
                ifs >> nx >> ny;
                ifs.close();
                rect.set_coord(nx, ny);
                std::ofstream ofs{"output.txt"};
                ofs << rect.center().x << ' ' << rect.center().y;
                ofs.close();
            },
            [&ifs]()
            {
                Rectangle rect;
                ifs >> rect;
                double new_size;
                ifs >> new_size;
                ifs.close();
                std::ofstream ofs{"output.txt"};
                ofs << rect.setw(new_size);
                ofs.close();
            },
            [&ifs]()
            {
                Rectangle rect;
                ifs >> rect;
                ifs.close();
                std::ofstream ofs{"output.txt"};
                ofs << rect.area();
                ofs.close();
            },
            [&ifs]()
            {
                Rectangle rect;
                ifs >> rect;
                ifs.close();
                std::ofstream ofs{"output.txt"};
                ofs << rect.perimeter();
                ofs.close();
            },
            [&ifs]()
            {
                Rectangle rec1, rec2;
                ifs >> rec1 >> rec2;
                ifs.close();
                std::ofstream ofs{"output.txt"};
                ofs << (rec1 + rec2).area();
                ofs.close();
            }};
        modes.at(mode)();
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}