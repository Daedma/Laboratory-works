//вариант 11
#include <iostream>
#include <cmath>
#include <array>
#include <iomanip>
#include <algorithm>
#include "table_print.hpp"
#include "sum.hpp"

//функция для подсчёта факториала числа
uint64_t factorial(uint64_t n) noexcept
{
    if (n == 0) return 1;
    uint64_t result = 1;
    for (uint64_t i = 1; i <= n; ++i)
    {
        result *= i;
    }
    return result;
}

//создает функцию для подсчета членов последовательности для данного X
auto create_an(long double x) noexcept
{
    return [x](size_t n) noexcept{
        static const auto pi = std::acos(-1.l);
        return (std::pow(2.l, n / 2.l) * std::sin((pi * n) / 4.l) * std::pow(x, n)) / factorial(n);//вариант 11
    };
}

//сообщить о некорректном вводе и привести std::cin в "порядок"
void badInputMessage()
{
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Oops... You enter invalid value, try again, if you want to get result\n>";
}

//функция для ввода значения с консоли с проверкой
template<typename T, typename... Func>
T getValue(Func... UnPred)//UnPred(Unary predicate) - функции для проверки правильности считанного значения
{
    T  value;
    while (!(std::cin >> value) || std::cin.peek() != '\n' || (!UnPred(value) || ...))
    {
        badInputMessage();
    }
    return value;
}

//Функция-обертка, вызывает getValue<double>
template<typename... Func>
long double getld(Func&&... UnPred)
{
    return getValue<long double>(std::forward<Func>(UnPred)...);
}

//Функция-обертка, вызывает getValue<std::string>
template<typename... Func>
std::string getstr(Func&&... UnPred)
{
    return getValue<std::string>(std::forward<Func>(UnPred)...);
}

//Узнает о намерениях пользователя о продолжении работы программы через консоль
bool keep_on()
{
    static const std::array<std::string, 10> choices = { "Yes", "yes", "y", "Y", "YES", "NO", "No", "no", "n", "N" };//допустимые ответы
    std::array<std::string, 10>::const_iterator find_result;//указывает на введенный пользователем допустимый ответ
    auto valid = [&find_result](const std::string& str){
        return (find_result = std::find(choices.cbegin(), choices.cend(), str)) != choices.cend();//если строка не соответсвует формату
    };
    std::cout << "Do you want to continue?\n>";
    getstr(valid);
    return  find_result < choices.cbegin() + choices.size() / 2;//положительные ответы содержаться в первой половине массива
}

//Получить параметр альфа (целое положительное число или положительное число с плавающей запятой)
std::variant<size_t, Sum::value_type> getAlpha()
{
    size_t pointCount = 0;//счётчик точек
    std::string num = getstr([&pointCount](const std::string& rhs){
        static const std::string numbers { "1234567890." };
        return rhs.find_first_not_of(numbers) == std::string::npos &&
            (pointCount = std::count(rhs.cbegin(), rhs.cend(), '.')) < 2 &&
            (pointCount ? std::stold(rhs) > 0.L : true);
        });
    if (pointCount)//если в строке есть точка, то вернуть long double
        return std::stold(num);
    return std::stoull(num);//иначе size_t
}

//вывести результаты в виде красивой таблички
void print_results(size_t nIteration, Sum::value_type LastMember, Sum::value_type PartialSum, Sum::value_type Precision)
{
    print_head({ { "Iteration number", 37 }, { tostr(nIteration), 25 } });
    print_line({ { "The last summed term of the series", 37 }, { tostr(LastMember), 25 } });
    print_line({ { "Current partial amount", 37 }, { tostr(PartialSum), 25 } });
    print_line({ { "Calculation accuracy", 37 }, { tostr(Precision), 25 } });
    print_end({ 37, 25 });
}

int main()
{
    std::cout << "================Calculating the sum of a series================\n\n";
    Sum::Cache cache { ".\\SUM.CACHE" };//кэш с ранее полученными результатами
    std::cout << std::setprecision(16);//установим точность для вывода
    do
    {
        std::cout << "Please, enter parameters:\nX = ";
        const Sum::value_type X = getld();//считываем параметр X
        std::cout << "Alpha" << " = ";
        auto alpha = getAlpha();//считываем параметр alpha
        if (std::holds_alternative<size_t>(alpha) && !std::get<size_t>(alpha))//если в alpha целое число и оно равно нулю
            std::cout << "Null iteration = no result\n";
        else if (!X)//если X равен нулю
        {
            print_results(1, 1, 1, 0);
        }
        else
        {
            Sum sum { create_an(X) };//представляет собой частичную сумму
            size_t nIteration;//кол-во итераций
            Sum::value_type LastMember,//последний просуммированный член
                PartialSum,//частичная сумма
                Precision;//точность вычисления
            if (cache.contains({ X, alpha }))//если результат уже содержится в кэше
            {
                const auto& result = cache.get({ X, alpha });
                nIteration = std::get<0>(result);
                LastMember = std::get<1>(result);
                PartialSum = std::get<2>(result);
                Precision = std::get<3>(result);
            }
            else//иначе вычислим частичную сумму
            {
                if (std::holds_alternative<size_t>(alpha))//если в alpha целое число (число итераций)
                    sum.calc(0ULL, std::get<size_t>(alpha) - 1);
                else//если в alpha дробное число (точность вычисления)
                    sum.calc(0ULL, std::get<Sum::value_type>(alpha));
                cache.add({ X, alpha }, sum);//загрузим результаты вычисления в кэш
                nIteration = sum.get_range().second;
                LastMember = sum.get_last();
                PartialSum = sum.get_sum();
                Precision = sum.get_precision();
            }
            print_results(nIteration, LastMember, PartialSum, Precision);//выведем результаты вычисления
        }
    } while (keep_on());//пока пользователь желает пользоваться программой
}