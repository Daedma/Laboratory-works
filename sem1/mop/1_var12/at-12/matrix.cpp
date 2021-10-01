/*
Задача:
За один проход двумерного массива найти мксимальное и минимальное значения. 
Входные данные: размеры матрицы и матрица. 
Выходные данные: минимальное и максимальное значения.
*/

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>

int main()
{
    using value_type = long double;
    uint32_t nlines, ncolumns;
    std::cout << "Please, nter the number of lines: ";
    while (std::cin.peek() == '-' || !(std::cin >> nlines) || std::cin.peek() != '\n')
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Incorrect input! Please, try again: ";
    }
    std::cout << "Please, nter the number of columns: ";
    while (std::cin.peek() == '-' || !(std::cin >> ncolumns) || std::cin.peek() != '\n')
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Incorrect input! Please, try again: ";
    }
    std::vector<std::vector<value_type>> matrix;
    matrix.reserve(nlines);
    for (uint32_t i = 0; i != nlines; ++i)
    {
        bool valid = true;
        do
        {
            std::cout << "Enter " << i + 1 << "st line: ";
            std::stringstream ss;
            std::string rdline;
            std::getline(std::cin, rdline);
            ss.str(rdline);
            std::vector<value_type> line;
            value_type t;
            for (uint32_t j = 0; j != ncolumns && valid; ++j)
            {
                if (ss >> t)
                    line.emplace_back(t);
                else
                    valid = false;
            }
            if (ss.str().find_first_not_of(' ', ss.tellg()) != std::string::npos)
                valid = false;
            if (valid)
                matrix.emplace_back(std::move(line));
            else
                std::cout << "Oops! You entered something wrong. Try again.\n";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } while (!valid);
    }

    value_type min, max;
    if (!matrix.empty() && matrix[0].size() >= 2)
    {
        min = std::min(matrix[0][0], matrix[0][1]);
    }
    for (auto i = matrix.cbegin(); i != matrix.cend(); ++i)
    {
        for (auto j = i->cbegin(); j != i->cend(); ++i)
        {
            if (*j < min) min = *j;
            if (*j > max) max = *j;
        }
    }
}