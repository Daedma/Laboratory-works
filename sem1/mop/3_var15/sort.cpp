/*
Вариат 15	
Б	
Строго случайные данные и случайные данные с малым числом уникальных значений
*/
#include <iostream>
#include <vector>
#include <iterator>
#include <array>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <functional>

#define UPPER_BOUND 10000
#define LOWER_BOUND -10000

//Class-timer
struct timer
{
private:
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<clock_t> m_beg;

public:
    timer() : m_beg(clock_t::now())
    {}

    void reset()
    {
        m_beg = clock_t::now();
    }

    double elapsed() const
    {
        return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
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


namespace my{
    template<typename OutputIt>
    void generate_n(OutputIt dest, size_t count, int lower_bound, int upper_bound) noexcept
    {
        static std::default_random_engine e { std::random_device {}() };
        std::uniform_int_distribution<int> d { lower_bound, upper_bound };
        for (size_t i = 0; i != count; ++i, ++dest)
            *dest = d(e);
    }

    template<typename OutputIt>
    void generate_n(OutputIt dest, size_t count, size_t unique_count, int lower_bound, int upper_bound)
    {

        std::vector<int> num_pool(unique_count);
        my::generate_n(num_pool.begin(), unique_count, lower_bound, upper_bound);
        static std::default_random_engine e { std::random_device {}() };
        std::uniform_int_distribution<size_t> d { 0, unique_count - 1 };
        for (size_t i = 0; i != count; ++i, ++dest)
            *dest = num_pool[d(e)];
    }

    template<typename BidirIt, typename Compare = std::less<>>
    BidirIt partition(BidirIt first, BidirIt last, Compare comp = std::less<> {})
    {
        using std::swap;
        auto p = std::prev(last);
        auto i = first;
        for (auto j = first; j != p; ++j)
            if (comp(*j, *p))
                swap(*i++, *j);
        swap(*i, *p);
        return i;
    }

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

void fill_vec(std::vector<int>& vec)
{
    std::cout << "Enter number of elements\n>";
    int count = geti(is_positive<>);
    std::cout << "Do you want enter value of elements(1) or generate(2)?\n>";
    int choice = geti([](int val){return val == 1 || val == 2; });
    if (choice == 1)
        input_container(std::back_inserter(vec), count);
    else
    {
        std::cout << "1 - strictly random data\nor\n2 - random data with a small number of unique values\n> ";
        choice = geti([](int val){return val == 1 || val == 2; });
        if (choice == 1)
            my::generate_n(std::back_inserter(vec), count, -count, count);
        else
            my::generate_n(std::back_inserter(vec), count, std::sqrt(count), -count, count);
    }
}

int main()
{
    static const size_t outCup = 20;
    std::cout << "==============| Fast sort |==============\n";
    do
    {
        std::vector<int> vec;
        fill_vec(vec);
        double mysort_time, stdsort_time;
        std::vector<int> mysort_vec = vec;
        std::vector<int> stdsort_vec = vec;

        timer t;
        my::sort(mysort_vec.begin(), mysort_vec.end());
        mysort_time = t.elapsed();

        t.reset();
        std::sort(stdsort_vec.begin(), stdsort_vec.end());
        stdsort_time = t.elapsed();

        size_t outCount = std::min(outCup, mysort_vec.size());
        std::cout << "Initial vector:\n";
        std::copy_n(vec.begin(), outCount, std::ostream_iterator<int>{std::cout, " "});
        std::cout << "\nResults of my implementation of the sorting algorithm:\n";
        std::copy_n(mysort_vec.cbegin(), outCount, std::ostream_iterator <int>{std::cout, " "});
        std::cout << "\nTime: " << mysort_time << " seconds\n";
        std::cout << "Results of the standart sorting algorithm:\n";
        std::copy_n(stdsort_vec.cbegin(), outCount, std::ostream_iterator <int>{std::cout, " "});
        std::cout << "\nTime: " << stdsort_time << " seconds\n";
    } while (keep_on());
}