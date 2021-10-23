#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <variant>
#include <array>
#include <sstream>
#include <vector>

using value_type = long double;

//преобразование любого типа в строку
template<typename T>
std::string tostr(T val)
{
    std::ostringstream oss;
    oss << std::setprecision(16) << val;
    return oss.str();
}

/*функции для печати таблицы*/
void print_head(std::initializer_list<std::pair<std::string, size_t>> Columns)
{
    static const char hline { '*' }, vline { '|' }, langle { '*' }, rangle { '*' }, delim { '*' };
    std::cout << std::setfill(hline) << langle;
    for (auto i = Columns.begin(); i != Columns.end(); ++i)
    {
        std::cout << std::setw(i->second + 1) << (i + 1 == Columns.end() ? rangle : delim);
    }
    std::cout << std::setfill(' ') << '\n';
    for (const auto& i : Columns)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_line(std::initializer_list<std::pair<long double, size_t>> Line)
{
    static const char vline { '|' };
    for (const auto& i : Line)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_line(std::initializer_list<std::pair<std::string, size_t>> Line)
{
    static const char vline { '|' };
    for (const auto& i : Line)
    {
        std::cout << vline << std::setw(i.second) << i.first;
    }
    std::cout << vline << '\n';
}

void print_end(std::initializer_list<size_t> Columns)
{
    static const char hline { '*' }, langle { '*' }, rangle { '*' }, delim { '*' };
    std::cout << std::setfill(hline) << langle;
    for (auto i = Columns.begin(); i != Columns.end(); ++i)
    {
        std::cout << std::setw(*i + 1) << (i + 1 == Columns.end() ? rangle : delim);
    }
    std::cout << std::setfill(' ') << '\n';
}

//вывести результаты в виде красивой таблички
void print_results(size_t nIteration, value_type LastMember, value_type PartialSum, value_type Precision)
{
    print_head({ { "Iteration number", 37 }, { tostr(nIteration), 25 } });
    print_line({ { "The last summed term of the series", 37 }, { tostr(LastMember), 25 } });
    print_line({ { "Current partial amount", 37 }, { tostr(PartialSum), 25 } });
    print_line({ { "Calculation accuracy", 37 }, { tostr(Precision), 25 } });
    print_end({ 37, 25 });
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
        return (find_result = std::find(choices.cbegin(), choices.cend(), str)) != choices.cend();//если строка соответсвует формату
    };
    std::cout << "Do you want to continue?\n>";
    getstr(valid);
    return  find_result < choices.cbegin() + choices.size() / 2;//положительные ответы содержаться в первой половине массива
}

//Получить параметр альфа (целое положительное число или положительное число с плавающей запятой)
std::variant<uint64_t, value_type> getAlpha()
{
    size_t pointCount = 0;//счётчик точек
    std::string num = getstr([&pointCount](const std::string& rhs){
        static const std::string numbers { "1234567890." };
        return rhs.find_first_not_of(numbers) == std::string::npos &&
            rhs.size() < 12 &&
            (pointCount = std::count(rhs.cbegin(), rhs.cend(), '.')) < 2 &&
            (pointCount ? std::stold(rhs) > 0.L : true);
        });
    if (pointCount)//если в строке есть точка, то вернуть long double
        return std::stold(num);
    return std::stoull(num);//иначе uint64_t
}

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

//Функция для вычисления члена последовательности  по его номеру и X
value_type SumMember(uint64_t Number, value_type X) noexcept
{
    static const auto pi = std::acos(-1.l);
    return (std::pow(2.l, Number / 2.l) * std::sin((pi * Number) / 4.l) * std::pow(X, Number)) / factorial(Number);
}

//Функция для подсчёта частичной суммы. Возвращает значения частичной суммы и последнего просуммированного элемента
std::pair<value_type, value_type> SumN(value_type X, uint64_t nIteration, bool Print)
{
    static std::map<value_type, std::vector<std::pair<value_type, value_type>>> cache;//кэш, куда записываются все произведенные вычисления
    static const std::array<size_t, 4> columns = { 4, 25, 25, 25 };//размеры колонок
    value_type PartialSum = 0,//текущая частичная сумма
        CurMember;//текущий суммированый член последовательности
    uint64_t start = 0;//номер, с которого стоит просуммировать члены последовательности
    auto pX = cache.emplace(X, std::vector<std::pair<value_type, value_type>>{});
    auto& refX = pX.first->second;//для краткости
    if (!pX.second)//если вычисления с данным параметром уже проводились
    {
        if (Print)//вывод сообщения о том, что данные были подгружены из кэша
            print_line({ { "**", columns[0] },
                { "upload", columns[1] },
                { "from", columns[2] },
                { "cache", columns[3] } });
        if (refX.size() >= nIteration)//если вектор содержит результаты для данного номера
            return refX[nIteration - 1];
        start = refX.size();//начнём суммировать члены с последнего вычисленного
        PartialSum = refX[start - 1].first;
    }
    CurMember = SumMember(++start, X);//найдем следующий член последовательности
    PartialSum += CurMember;//прибавим его к текущей частичной сумме
    refX.emplace_back(PartialSum, CurMember);//добавим результат в кэш
    for (uint64_t i = start + 1; i <= nIteration + 1; ++i)
    {
        CurMember = SumMember(i, X);
        PartialSum += CurMember;
        refX.emplace_back(PartialSum, CurMember);
        if (Print)//вывод результата предыдущего вычисления
            print_line({ { i - 1, columns[0] },
                { refX[i - 2].second, columns[1] },
                { refX[i - 2].first, columns[2] },
                { std::abs(refX[i - 1].second / refX[i - 2].first), columns[3] } });
    }
    return refX[nIteration - 1];
}

//Подсчет и вывод результатов на экран для целочисленного параметра Alpha
void calc(value_type X, uint64_t N)
{
    static const std::array<size_t, 4> columns = { 4, 25, 25, 25 };//размеры колонок
    print_head({ { "n", columns[0] }, { "an", columns[1] }, { "Sn", columns[2] }, { "ALPHAn", columns[3] } });
    auto Cur = SumN(X, N, true);
    print_end({ columns[0], columns[1], columns[2], columns[3] });
    auto Next = SumN(X, N + 1, false);
    std::cout << "Results:\n";
    print_results(N, Cur.second, Cur.first, std::abs(Next.second / Cur.first));
}

//Подсчет и вывод результатов на экран для дробного параметра Alpha
void calc(value_type X, value_type Alpha)
{
    static const std::array<size_t, 4> columns = { 4, 25, 25, 25 };//размеры колонок
    value_type CurPrecision;//текущая точность вычисления
    uint64_t N = 1;//текущий номер члена последовательности
    print_head({ { "n", columns[0] }, { "an", columns[1] }, { "Sn", columns[2] }, { "ALPHAn", columns[3] } });
    std::pair<value_type, value_type> Cur,//текущий
        Next = SumN(X, N, true);//и следующий просуммированные члены
    do
    {
        Cur = Next;
        Next = SumN(X, ++N, true);
        CurPrecision = std::abs(Next.second / Cur.first);
    } while (CurPrecision > Alpha);
    print_end({ columns[0], columns[1], columns[2], columns[3] });
    print_results(N - 1ULL, Cur.second, Cur.first, CurPrecision);
}

int main()
{
    std::cout << "================Calculating the sum of a series================\n\n";
    std::cout << std::setprecision(16);//установим точность для вывода
    do
    {
        std::cout << "Please, enter parameters:\nX = ";
        value_type X = getld();
        std::cout << "Alpha = ";
        auto alpha = getAlpha();
        if (std::holds_alternative<uint64_t>(alpha))//если пользователь ввёл целочисленное число
            calc(X, std::get<uint64_t>(alpha));
        else//если дробное
            calc(X, std::get<value_type>(alpha));
    } while (keep_on());
}