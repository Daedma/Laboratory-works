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
    using value_type = int;
    char choice;
    do
    {
        uint32_t nlines, ncolumns;
        std::cout << "Please, enter the number of lines: ";
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
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        for (uint32_t i = 0; i != nlines; ++i)
        {
            bool valid;
            do
            {
                valid = true;
                std::cout << "Enter " << i + 1 << "st line: ";
                std::stringstream ss;
                std::string rdline;
                std::getline(std::cin, rdline, '\n');
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
            } while (!valid);
        }

        value_type min, max;
        bool empty = false;
        if (nlines != 0 && ncolumns >= 2)
        {
            min = std::min(matrix[0][0], matrix[0][1]);
            max = std::max(matrix[0][0], matrix[0][1]);
        }
        else if (ncolumns != 0 && nlines >= 2)
        {
            min = std::min(matrix[0][0], matrix[1][0]);
            max = std::max(matrix[0][0], matrix[1][0]);
        }
        else if (nlines == 1 && ncolumns == 1)
            min = max = matrix[0][0];
        else
            empty = true;
        if (!empty)
        {
            for (auto i = matrix.cbegin(); i != matrix.cend(); ++i)
            {
                for (auto j = i->cbegin(); j != i->cend(); ++j)
                {
                    if (*j < min)
                        min = *j;
                    else if (*j > max)
                        max = *j;
                }
            }
        }

        if (empty)
            std::cout << "Your matrix is empty.";
        else if (nlines == 1 && ncolumns == 1)
            std::cout << "Your matrix contains only one element: " << max;
        else if (max == min)
            std::cout << "Elements in your matrix is equal: " << max;
        else
            std::cout << "Maximum value is: " << max
            << "\nMinimum value is: " << min;
        std::cout << '\n';
        std::cout << "Do you want to continue? (Y/N): ";
        std::cin >> choice;
        while (choice != 'Y' && choice != 'N' || std::cin.peek() != '\n') //пока не будут переданы правильные значения в choice
        {
            std::cout << "Incorrect input! Please try again: ";                 //приглашение к повторному вводу
            std::cin.clear();                                                   //сбросим флаги ошибок
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
            std::cin >> choice;
        }
    } while (choice == 'Y');
}