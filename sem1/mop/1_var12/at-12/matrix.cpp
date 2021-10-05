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
    using value_type = int;//тип значений в матрице
    char choice;//выбор пользователя
    do//основной цикл программы
    {
        int64_t nlines, ncolumns;//количество строк и столбцов
        std::cout << "Please, enter the number of lines: ";//приглашение пользователя к вводу количества строк
        while (!(std::cin >> nlines) || nlines < 0 || std::cin.peek() != '\n')//пока ввод не будет удачен
        {
            std::cin.clear();//сбросить флаг ошибки
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');//очистить буфер
            std::cout << "Incorrect input! Please, try again: ";//приглашение к повторному вводу
        }
        std::cout << "Please, nter the number of columns: ";//приглашение пользователя к вводу количества столбцов
        while (!(std::cin >> ncolumns) || ncolumns < 0 || std::cin.peek() != '\n')//пока ввод не будет удачен
        {
            std::cin.clear();//сбросить флаг ошибки
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');//очистить буфер
            std::cout << "Incorrect input! Please, try again: ";//приглашение к повторному вводу
        }
        std::vector<std::vector<value_type>> matrix;//матрица
        matrix.reserve(nlines);//зарезервируем сразу необходимое количество памяти для более быстрой вставки элементов в вектор
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');//очистить буффер

        //ввод матрицы
        for (uint32_t i = 0; i != nlines; ++i)
        {
            bool valid;//переменная для выхода из цикла при правильном вводе строки
            do
            {
                valid = true;
                std::cout << "Enter " << i + 1 << "st line: ";//приглашение к вводу строки
                std::stringstream ss;
                std::string rdline;//считываемая строка
                std::getline(std::cin, rdline);//считывание строки до перевода строки
                ss.str(rdline);//свяжем ss со считанной строкой
                std::vector<value_type> line;//в этот вектор будут записаны считанные значения
                value_type t;//текущая считанная переменная
                for (uint32_t j = 0; j != ncolumns && valid; ++j)
                {
                    if (ss >> t)//если чтение произошло успешно, то добавить переменную в вектор
                        line.emplace_back(t);
                    else//если нет, то записать в valid информацию о том, что произошла ошибка
                        valid = false;
                }
                if (valid && ss.str().find_first_not_of(' ', ss.tellg()) != std::string::npos)//если в считанной строке остались символы
                    valid = false;
                if (valid)//если ошибок не было, то добавить строку в матрицу
                    matrix.emplace_back(std::move(line));
                else
                    std::cout << "Oops! You entered something wrong. Try again.\n";//сообщить пользователю о некорректном вводе
            } while (!valid);//пока не будет введена правильная строка
        }

        value_type min, max;//минимальные и максимальные значения
        bool empty = matrix.empty();//матрица пуста
        //выберем начальные значения для min и max
        if (!empty)
        {
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
            for (auto i = matrix.cbegin(); i != matrix.cend(); ++i)//сам поиск максимального и минимального значений
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
        //вывод результатов
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
        std::cout << "Do you want to continue? (Y/N): ";//спрашиваем у пользователя, желает ли он продолжить пользоваться нашей программой
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