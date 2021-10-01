/*
Задание:
Оценка характеристик поседовательности чисел. 
Входные данные: последовательность поожительных действительных чисел. 
Выходные данные: максимальное, минимальное и среднее арифметическое значения последовательности. 
Особенности: ввод последовательности заканчивается при вводе 0.
*/

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iomanip>

int main()
{
    char choice;
    do
    {
        std::vector<long double> row;
        long double num { 0.0L };
        bool status = true, error = false;
        std::cout << "Please, enter positive real numbers (put zero at the end of the sequence you entered): ";
        //ввод
        while (status)
        {
            if (std::cin >> num && num >= 0.0L)
            {
                if (num == 0.0L)
                    status = false;
                else
                    row.push_back(num);
            }
            else
            {
                std::cin.clear();                                                   //сбросим флаги ошибок
                status = false;
                error = true;
            }
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
        //вывод
        if (!error)
        {
            if (row.size())
            {
                auto min_max = std::minmax_element(row.cbegin(), row.cend());
                std::cout << "In the sequence you entered: \n";
                if (min_max.first == min_max.second)
                    std::cout << "all elements are equal,\n";
                else
                    std::cout << "minimum element value is " << *(min_max.first) << ",\n"
                    << "maximum element value is " << *(min_max.second) << ",\n";
                std::cout << "arithmetic mean is " << std::accumulate(row.cbegin(), row.cend(), 0.0L) / row.size()
                    << ".\n\n";
            }
            else
                std::cout << "The sequence you entered contains no elements.\n\n";
            std::cout << "Do you want to continue?(Y - Yes/N - No): ";
            std::cin >> choice;
            while (choice != 'Y' && choice != 'N' || std::cin.peek() != '\n') //пока не будут переданы правильные значения в choice
            {
                std::cout << "Incorrect input! Please try again: ";                  //приглашение к повторному вводу
                std::cin.clear();                                                   //сбросим флаги ошибок
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
                std::cin >> choice;
            }
        }
        else
        {
            std::cout << "Oops! You entered something wrong. Try again.\n";
            choice = 'Y';
        }

    } while (choice == 'Y');
    std::cout << "Good bye!\n";
}