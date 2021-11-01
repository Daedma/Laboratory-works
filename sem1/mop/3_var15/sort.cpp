#include <iostream>
#include <vector>
#include <iterator>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <functional>


//Class-timer
struct timer
{
private:
    using clock_t = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock_t> m_beg;

public:
    timer() : m_beg(clock_t::now())
    {}

    //сбросить таймер
    void reset()
    {
        m_beg = clock_t::now();
    }
    //получить пройденное с момента отсчёта  время
    long double elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::duration<long double, std::milli>>(clock_t::now() - m_beg).count();
    }
};

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

//Функция-обертка, вызывает getValue<int>
template<typename... Func>
int geti(Func&&... UnPred)
{
    return getValue<int>(std::forward<Func>(UnPred)...);
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

//пространство с моими реализациями функций из стандартной библиотеки
namespace my{
    //заполнить контейнер случайными значениями из заданного диапазона 
    template<typename OutputIt, typename T>
    void generate_n(OutputIt dest, size_t count, T lower_bound, T upper_bound) noexcept
    {
        static std::default_random_engine e { std::random_device {}() };
        std::uniform_int_distribution<T> d { lower_bound, upper_bound };
        for (size_t i = 0; i != count; ++i, ++dest)
            *dest = d(e);
    }

    //заполнить контейнер случайными значениями с малым числом уникальных
    template<typename OutputIt, typename T>
    void generate_n(OutputIt dest, size_t count, size_t unique_count, T lower_bound, T upper_bound)
    {

        std::vector<T> num_pool(unique_count);
        my::generate_n(num_pool.begin(), unique_count, lower_bound, upper_bound);
        static std::default_random_engine e { std::random_device {}() };
        std::uniform_int_distribution<size_t> d { 0, unique_count - 1 };
        for (size_t i = 0; i != count; ++i, ++dest)
            *dest = num_pool[d(e)];
    }

    //разделить массив 
    template<typename BidirIt, typename Compare = std::less<>>
    BidirIt partition(BidirIt first, BidirIt last, Compare comp = std::less<> {})
    {
        using std::swap;
        auto p = std::prev(last);//опорный элемент
        auto i = first;//указывает на элементы большие либо равные p
        for (auto j = first; j != p; ++j)
            if (comp(*j, *p))
                swap(*i++, *j);
        swap(*i, *p);
        return i;
    }

    //быстрая сортировка
    template< typename BidirIt, typename Compare = std::less<>>
    void sort(BidirIt first, BidirIt last, Compare comp = std::less<> {})
    {
        if (std::distance(first, last) > 1)
        {
            auto bound = my::partition(first, last, comp);
            my::sort(first, bound, comp);
            my::sort(std::next(bound), last, comp);
        }
    }
}

//ввод массива вручную
template<typename OutputIt>
void input_container(OutputIt dest, size_t count)
{
    std::cout << "Enter " << count << " elements:\n";
    for (size_t i = 0; i != count; ++i, ++dest)
    {
        std::cout << (i + 1) << ": ";
        *dest = geti();
    }
}

template<typename T = int>
bool is_positive(T val) noexcept
{
    return val > 0;
}

//заполнить вектор различными способами, которыми пожелает пользователь
void fill_vec(std::vector<int>& vec)
{
    std::cout << "Enter number of elements\n>";
    int count = geti(is_positive<>);
    std::cout << "Do you want enter value of elements(1) or generate(2)?\n>";
    int choice = geti([](int val) noexcept{return val == 1 || val == 2; });
    if (choice == 1)
        input_container(std::back_inserter(vec), count);
    else
    {
        std::cout << "1 - strictly random data\nor\n2 - random data with a small number of unique values?\n> ";
        choice = geti([](int val) noexcept{return val == 1 || val == 2; });
        if (choice == 1)
            my::generate_n(std::back_inserter(vec), count, -count, count);
        else
            my::generate_n(std::back_inserter(vec), count, std::sqrt(count), -count, count);
    }
}

int main()
{
    static const size_t outCup = 20;//максимальное количество выводимых элементов
    std::cout << "==============| Fast sort |==============\n";
    do
    {
        std::vector<int> vec;//изначальный вектор
        fill_vec(vec);
        long double mysort_time, stdsort_time;
        std::vector<int> mysort_vec = vec;//вектор, отсортированный моим алгоритмом сортировки
        std::vector<int> stdsort_vec = vec;//вектор, отсортированный стандартным алгоритмом сортировки

        timer t;
        my::sort(mysort_vec.begin(), mysort_vec.end());
        mysort_time = t.elapsed();//фиксируем время

        t.reset();
        std::sort(stdsort_vec.begin(), stdsort_vec.end());
        stdsort_time = t.elapsed();//фиксируем время

        size_t outCount = std::min(outCup, mysort_vec.size());//количество выводимых символов
        std::cout << "Initial vector:\n";
        std::copy_n(vec.begin(), outCount, std::ostream_iterator<int>{std::cout, " "});//вывести первоначальный вектор

        std::cout << "\nResults of my implementation of the sorting algorithm:\n";
        std::copy_n(mysort_vec.cbegin(), outCount, std::ostream_iterator <int>{std::cout, " "});//вывести отсортированный вектор
        std::cout << "\nTime: " << mysort_time << " milliseconds\n";

        std::cout << "Results of the standart sorting algorithm:\n";
        std::copy_n(stdsort_vec.cbegin(), outCount, std::ostream_iterator <int>{std::cout, " "});//вывести отсортированный вектор
        std::cout << "\nTime: " << stdsort_time << " milliseconds\n";

    } while (keep_on());
}