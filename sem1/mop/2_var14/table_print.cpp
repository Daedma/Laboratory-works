#include "table_print.hpp"
#include <iostream>
#include <iomanip>

//функции для печати таблицы
void print_head(std::initializer_list<std::pair<std::string, size_t>> Columns)
{
    static const char hline { char(196) }, vline { char(179) }, langle { char(218) }, rangle { char(191) }, delim { char(194) };
    std::cout << std::setfill(hline) << langle;
    for (auto i = Columns.begin(); i != Columns.end(); ++i)
    {
        std::cout << std::setw(i->second + 1) << (i + 1 == Columns.end() ? rangle : delim);
    }
    std::cout << std::setfill(' ') << '\n';
    for (const auto& i : Columns)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_line(std::initializer_list<std::pair<long double, size_t>> Line)
{
    static const char vline { char(179) };
    for (const auto& i : Line)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_line(std::initializer_list<std::pair<std::string, size_t>> Line)
{
    static const char vline { char(179) };
    for (const auto& i : Line)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_end(std::initializer_list<size_t> Columns)
{
    static const char hline { char(196) }, langle { char(192) }, rangle { char(217) }, delim { char(193) };
    std::cout << std::setfill(hline) << langle;
    for (auto i = Columns.begin(); i != Columns.end(); ++i)
    {
        std::cout << std::setw(*i + 1) << (i + 1 == Columns.end() ? rangle : delim);
    }
    std::cout << std::setfill(' ') << '\n';
}