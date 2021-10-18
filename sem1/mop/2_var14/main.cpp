#include <iostream>
#include <cmath>
#include <array>
#include <iomanip>
#include <string>
#include <algorithm>
#include "sum.hpp"
#define INDEX_N char(252) 
#define ALPHA char(224)

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

auto create_an(long double x) noexcept
{
    return [x](size_t n) noexcept{
        return (factorial(factorial(2 * n)) * std::powl(x, 2 * n)) / factorial(factorial(2 * n + 1));
    };
}

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
    static const std::array<std::string, 8> choices = { "Yes", "yes", "y", "Y", "No", "no", "n", "N" };//допустимые ответы
    std::array<std::string, 8>::const_iterator find_result;//указывает на введенный пользователем допустимый ответ
    auto valid = [&find_result](const std::string& str){
        return (find_result = std::find(choices.cbegin(), choices.cend(), str)) != choices.cend();
    };
    std::cout << "Do you want to continue?\n>";
    getstr(valid);
    return  find_result < choices.cbegin() + choices.size() / 2;
}

std::variant<size_t, Sum::value_type> getAlpha()
{
    uint64_t nit;
    Sum::value_type alpha;
    while (true)
    {
        std::cin >> std::ws;
        if (std::cin.peek() == '-' || !(!std::cin >> nit && std::cin.peek() == '.') && !std::cin)
        {
            badInputMessage();
        }
        else if (std::cin.peek() == '.')
        {
            std::cin.clear();
            if (!(std::cin >> alpha) || std::cin.peek() != '\n' || alpha == 0)
            {
                badInputMessage();
            }
            else
                return alpha + nit;
        }
        else if (std::cin.peek() != '\n')
        {
            badInputMessage();
        }
        else
            return nit;
    }
    return 0ULL;
}

void print_results(size_t nIteration, Sum::value_type LastMember, Sum::value_type PartialSum, Sum::value_type Precision)
{
    size_t max_width = std::max({ static_cast<Sum::value_type>(nIteration), LastMember, PartialSum, Precision },
        [](Sum::value_type lhs, Sum::value_type rhs){
            return std::to_string(lhs).size() > std::to_string(rhs).size();
        });
    short max_name = 37;
    std::cout << char(218) << std::setfill(char(196)) << std::setw(max_name) << char(194) << std::setw(max_width + 1) << char(191) << '\n' << std::setfill(' ');
    std::cout << char(179) << "Iteration number" << std::setw(max_name - 17) << char(179) << std::setw(max_width + 1) << nIteration << char(179) << '\n';
    std::cout << char(179) << "The last summed term of the series" << std::setw(max_name - 35) << char(179) << std::setw(max_width + 1) << LastMember << char(179) << '\n';
    std::cout << char(179) << "Current partial amount" << std::setw(max_name - 23) << char(179) << std::setw(max_width + 1) << PartialSum << char(179) << '\n';
    std::cout << char(179) << "Calculation accuracy" << std::setw(max_name - 21) << char(179) << std::setw(max_width + 1) << Precision << char(179) << '\n';
    std::cout << char(192) << std::setfill(char(196)) << std::setw(max_name) << char(193) << std::setw(max_width + 1) << char(217) << '\n' << std::setfill(' ');
}

int main()
{
    std::cout << "================Calculating the sum of a series================\n\n";
    Sum::Cache cache { ".\\SUM.CACHE" };
    do
    {
        std::cout << "Please, enter parameters:\nX = ";
        const Sum::value_type X = getld([](Sum::value_type val){return val > -1 && val < 1; });
        std::cout << ALPHA << " = ";
        auto alpha = getAlpha();
        if (!X)
        {
            print_results(0, 0, 0, 0);
        }
        else
        {
            Sum sum { create_an(X) };
            size_t nIteration;
            Sum::value_type LastMember, PartialSum, Precision;
            if (cache.contains({ X, alpha }))
            {
                const auto& result = cache.get({ X, alpha });
                nIteration = std::get<0>(result);
                LastMember = std::get<1>(result);
                PartialSum = std::get<2>(result);
                Precision = std::get<3>(result);
            }
            else
            {
                if (std::holds_alternative<size_t>(alpha))
                    sum.calc(std::get<size_t>(alpha));
                else
                    sum.calc(std::get<Sum::value_type>(alpha));
                cache.add({ X, alpha }, sum);
                const auto& result = cache.get({ X, alpha });
                nIteration = sum.get_range().second;
                LastMember = sum.get_last();
                PartialSum = sum.get_sum();
                Precision = sum.get_precision();
            }
            print_results(nIteration, LastMember, PartialSum, Precision);
        }
    } while (keep_on());
}