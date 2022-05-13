#include <stdexcept>
#include <fstream>
#include <iostream>

class Rectangle //Класс прямоугольника
{
    double _width, _height; //ширина, высота
    struct Point
    {
        double x, y;
    } _center; //координаты центра

public:
    Rectangle() noexcept : _width{0}, _height{0}, _center{0, 0} {}                                               //конструктор по умолчанию
    Rectangle(const Rectangle &rhs) noexcept : _width{rhs._width}, _height{rhs._height}, _center{rhs._center} {} //конструктор копирования
    Rectangle(double width, double height, Point center) : _width{width}, _height{height}, _center{center}       //конструктор инициализации объекта
    {
        if (width < 0 || height < 0)
            throw std::invalid_argument{"ERROR: rectangle sides cannot be negative"};
    }
    ~Rectangle() noexcept {}                            //Деструктор
    Rectangle(Rectangle &&) = default;                  //Конструктор перемещения
    Rectangle &operator=(Rectangle &&) = default;       //Оператор присваивания перемещением
    Rectangle &operator=(const Rectangle &rhs) noexcept //Оператор присваивания
    {
        _width = rhs._width;
        _height = rhs._height;
        _center = rhs._center;
        return *this;
    }

    //Операции сравнения
    bool operator<(const Rectangle &rhs) const noexcept { return area() < rhs.area(); }
    bool operator<=(const Rectangle &rhs) const noexcept { return area() <= rhs.area(); }
    bool operator>(const Rectangle &rhs) const noexcept { return area() > rhs.area(); }
    bool operator>=(const Rectangle &rhs) const noexcept { return area() >= rhs.area(); }
    bool operator==(const Rectangle &rhs) const noexcept
    {
        return _width == rhs._width &&
               _height == rhs._height &&
               _center.x == rhs._center.x &&
               _center.y == rhs._center.y;
    }
    bool operator!=(const Rectangle &rhs) const noexcept { return !(*this == rhs); }

    double width() const noexcept { return _width; }
    double height() const noexcept { return _height; }

    double setw(double width) //Изменить ширину
    {
        if (width < 0)
            throw std::invalid_argument{"ERROR: width cannot be negative"};
        return _width = width;
    }
    double seth(double height) //Изменить высоту
    {
        if (height < 0)
            throw std::invalid_argument{"ERROR: height cannot be negative"};
        return _height = height;
    }
    void set_size(double width, double height) //Изменить размеры
    {
        *this = Rectangle{width, height, _center};
    }
    Point &set_coord(double x, double y) noexcept //Изменить координаты центра
    {
        _center.x = x;
        _center.y = y;
        return _center;
    }
    Point &center() noexcept { return _center; } //Прямой доступ к координатам центра
    const Point &center() const noexcept { return _center; }

    double area() const noexcept { return _width * _height; }            //Вычисление площади
    double perimeter() const noexcept { return (_width + _height) * 2; } //Вычисление периметра
    Rectangle operator+(const Rectangle &rhs) const noexcept             //Сложение прямоугольников
    {
        return {_width + rhs._width, _height + rhs._height, {(_center.x + rhs._center.x) / 2, (_center.y + rhs._center.y) / 2}};
    }
};

std::ostream &operator<<(std::ostream &os, const Rectangle &rhs) //Оператор вывода в поток
{
    return os << rhs.center().x << ' ' << rhs.center().y << ' ' << rhs.width() << ' ' << rhs.height();
}

std::istream &operator>>(std::istream &is, Rectangle &rhs) //Оператор чтения из потока
{
    double width, height, x, y;
    if (is >> x >> y >> width >> height)
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
    enum Modes
    {
        COMP,
        COORD,
        SIZE,
        AREA,
        PERIM,
        SUM
    };
    try
    {
        std::ifstream ifs{"input.txt"};                       //Открываем файл для чтения
        ifs.exceptions(std::ios::failbit | std::ios::badbit); //Включаем исключения у потока
        int mode;                                             //Режим работы программы
        ifs >> mode;
        switch (mode)
        {
        case COMP: //Сравнение прямоугольников
        {
            Rectangle rec1, rec2;
            ifs >> rec1 >> rec2;
            ifs.close();
            std::ofstream ofs{"output.txt"};
            ofs << (rec1 == rec2) << '\n';
            ofs << (rec1 != rec2) << '\n';
            ofs << (rec1 <= rec2) << '\n';
            ofs << (rec1 >= rec2) << '\n';
            ofs << (rec1 > rec2) << '\n';
            ofs << (rec1 < rec2) << '\n';
            ofs.close();
            break;
        }
        case COORD: //Изменение координаты
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
            break;
        }
        case SIZE: //Изменение размера одной из сторон
        {
            Rectangle rect;
            ifs >> rect;
            double new_size;
            ifs >> new_size;
            ifs.close();
            std::ofstream ofs{"output.txt"};
            ofs << rect.setw(new_size);
            ofs.close();
            break;
        }
        case AREA: //Вычисление площади прямоугольника
        {
            Rectangle rect;
            ifs >> rect;
            ifs.close();
            std::ofstream ofs{"output.txt"};
            ofs << rect.area();
            ofs.close();
            break;
        }
        case PERIM: //Вычисление периметра прямоугольника
        {
            Rectangle rect;
            ifs >> rect;
            ifs.close();
            std::ofstream ofs{"output.txt"};
            ofs << rect.perimeter();
            ofs.close();
            break;
        }
        case SUM: //Вычисление суммы прямоугольников
        {
            Rectangle rec1, rec2;
            ifs >> rec1 >> rec2;
            ifs.close();
            std::ofstream ofs{"output.txt"};
            ofs << (rec1 + rec2).area();
            ofs.close();
            break;
        }
        default:
            throw std::runtime_error{"ERROR: undefined mode"};
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}