#include <iostream>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <array>
#include <algorithm>
#include <iomanip>

long double calc_height(const long double PathPart) noexcept
{
    static constexpr long double half_g = 4.9L;
    return half_g * ((2.L - PathPart + 2.L * std::sqrtl(1.L - PathPart)) / std::powl(PathPart, 2.L));
}

template<typename T, typename... Func>
T getValue(Func... UnPred)
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

template<typename... Func>
long double getld(Func&&... UnPred)
{
    return getValue<long double>(std::forward<Func>(UnPred)...);
}

template<typename... Func>
std::string getstr(Func&&... UnPred)
{
    return getValue<std::string>(std::forward<Func>(UnPred)...);
}

bool keep_on()
{
    static const std::array<std::string, 8> choices = { "Yes", "yes", "y", "Y", "No", "no", "n", "N" };
    std::array<std::string, 8>::const_iterator find_result;
    auto valid = [&find_result](const std::string& str){
        return (find_result = std::find(choices.cbegin(), choices.cend(), str)) != choices.cend();
    };
    getstr(valid);
    return  find_result < choices.cbegin() + choices.size() / 2;
}

int main()
{
    std::cout << "=========Last second of the fall=========\n";
    auto valid_val = [](long double PathPart) noexcept{
        return PathPart > 0.L && PathPart < 1.L;
    };
    std::cout << std::setprecision(std::numeric_limits<long double>::max_digits10 + 1);
    do
    {
        std::cout << "Input a fraction of distance covered in the last second\n>";
        long double PathPart = getld(valid_val);
        std::cout << "Total height: " << calc_height(PathPart) << '\n';
        std::cout << "Do you want to continue?\n>";
    } while (keep_on());
}