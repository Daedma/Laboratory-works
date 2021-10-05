/*
Задание:
Оценка характеристик поседовательности чисел. 
Входные данные: последовательность поожительных действительных чисел. 
Выходные данные: максимальное, минимальное и среднее арифметическое значения последовательности. 
Особенности: ввод последовательности заканчивается при вводе 0.
*/

#include <iostream>
#include <limits>
#include <vector>

int main()
{
    char choice; //выбор пользователя о завершении/продолжении работы программы
    do           //основной цикл программы
    {
        std::vector<long double> row;                                                                           //ряд введенных пользователем чисел
        long double num{0.0L};                                                                                  //текущее введенное пользователем число
        bool status = true,                                                                                     //используется в цикле ввода для выхода из него в нужный момент
            error = false;                                                                                      //если true, то значит ввод был некорректен
        std::cout << "Please, enter positive real numbers (put zero at the end of the sequence you entered): "; //приглашение пользователя к вводу
        //ввод
        while (status) //пока пользователь не введет 0 или некорректное значение
        {
            if (std::cin >> num && num >= 0.0L) //если ввод произошел успешно
            {
                if (num == 0.0L) //если пользователь введет ноль, то прервать чтение
                {
                    status = false;                    //выйти из цикла ввода
                    error = (std::cin.peek() != '\n'); //если после 0 оказались посторонние значения
                }
                else
                    row.push_back(num); //иначе добавить в вектор
            }
            else //если ввод был неправильным
            {
                std::cin.clear(); //сбросим флаги ошибок
                status = false;   //выйти из цикла ввода
                error = true;     //записать в переменную информацию о том, что произошла ошибка
            }
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
        //вывод
        if (!error) //если ввод был успешным
        {
            if (row.size()) //если вектор не пустой
            {
                long double min, max;//минимальное и максимальное значения
                min = max = row[0];
                long double sum = 0;//сумма элементов вектора
                for (auto i : row)
                {
                    sum += i;
                    if (i > max)
                        max = i;
                    else if (i < min)
                        min = i;
                }
                std::cout << "In the sequence you entered: \n";
                if (min != max) //если все значения различны
                {
                    //выведем оценку характеристик последовательности чисел...
                    std::cout << "minimum element value is " << min << ",\n"  //минимальное значение
                              << "maximum element value is " << max << ",\n"; //максимальное значение
                    std::cout << "arithmetic mean is " << sum / row.size()    //среднее арифметическое
                              << ".\n\n";
                }
                else                                                         //если все значение одинаковы
                    std::cout << "all elements are equal: " << min << ".\n\n"; //сообщить об этом пользователю
            }
            else
                std::cout << "The sequence you entered contains no elements.\n\n"; //ряд чисел пуст
            std::cout << "Do you want to continue?(Y - Yes/N - No): ";
            std::cin >> choice;
            while (choice != 'Y' && choice != 'N' || std::cin.peek() != '\n') //пока не будут переданы правильные значения в choice
            {
                std::cout << "Incorrect input! Please try again: ";                 //приглашение к повторному вводу
                std::cin.clear();                                                   //сбросим флаги ошибок
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //очистим буфер
                std::cin >> choice;
            }
        }
        else
        {
            std::cout << "Oops! You entered something wrong. Try again.\n"; //Ошибка! Просим пользователя заново ввести данные
            choice = 'Y';
        }

    } while (choice == 'Y');
    std::cout << "Good bye!\n"; //Прощаемся с пользователем по окончанию программы
}