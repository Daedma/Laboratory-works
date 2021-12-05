#pragma once
#include <limits>
#include <iostream>
#include <string>
#include <array>
#include <algorithm>

namespace{
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
        std::cin.get();
        return value;
    }

    //Функция-обертка, вызывает getValue<std::string>
    template<typename... Func>
    std::string getstr(Func&&... UnPred)
    {
        return getValue<std::string>(std::forward<Func>(UnPred)...);
    }

    //Функция-обертка, вызывает getValue<int>
    template<typename... Func>
    int geti(Func&&... UnPred)
    {
        return getValue<int>(std::forward<Func>(UnPred)...);
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
}