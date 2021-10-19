#pragma once
#include <initializer_list>
#include <sstream>
#include <utility>
#include <string>
#include <iomanip>

void print_head(std::initializer_list<std::pair<std::string, size_t>>);
void print_line(std::initializer_list<std::pair<long double, size_t>>);
void print_line(std::initializer_list<std::pair<std::string, size_t>>);
void print_end(std::initializer_list<size_t>);

template<typename T>
std::string tostr(T val)
{
    std::ostringstream oss;
    oss << std::setprecision(16) << val;
    return oss.str();
}