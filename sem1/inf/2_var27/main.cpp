#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include <array>
#include <algorithm>
#include <iomanip>

//функция для подсчета высоты, с которой падало тело, по расстоянию, 
//пройденным телом за последнюю секунду падения
long double calc_height(const long double PathPart) noexcept
{
    static constexpr long double half_g = 4.9L;//ускорение свободного падения пополам
    return half_g * ((2.L - PathPart + 2.L * std::sqrtl(1.L - PathPart)) / std::powl(PathPart, 2.L));//формула
}

//функция для ввода значения с консоли с проверкой
template<typename T, typename... Func>
T getValue(Func... UnPred)//UnPred(Unary predicate) - функции для проверки правильности считанного значения
{
    T  value;
    while (!(std::cin >> value) || std::cin.peek() != '\n' || (!UnPred(value) || ...))
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Oops... You enter invalid value, try again, if you want to get result\n>";
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

int main()
{
    std::cout << "=========Last second of the fall=========\n";
    auto valid_val = [](long double PathPart) noexcept{//проверяет допустимость введенного пользователем значения для доли расстояния
        return PathPart > 0.L && PathPart < 1.L;
    };
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10 + 1);
    do
    {
        std::cout << "Input a fraction of distance covered in the last second\n>";
        if (const auto result = calc_height(getld(valid_val)); std::isnormal(result))//если подсчёт удался
            std::cout << "Total height: " << result << '\n';
        else
            std::cout << "Error! Failed to calculate the height.\n";
    } while (keep_on());
}