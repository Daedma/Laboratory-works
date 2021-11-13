/*
Вариат 14	
Замена подстроки в строке. 
Входные данные: исходная строка, строка, котору нужно заменить, и строка для вставки. 
Выходные данные: строка.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <stdexcept>
#include <string_view>
#include <sstream>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <array>
#include <vector>

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
    (void) std::cin.get();
    return value;
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

void replace_all(std::string& aSource, std::string_view aOld, std::string_view aNew)
{
    auto cur = aSource.find(aOld);
    while (cur != std::string::npos)
    {
        aSource.replace(cur, aOld.size(), aNew);
        cur = aSource.find(aOld, cur + aNew.size());
    }
}

template<typename Func>
std::string read_lines(std::istream& is, Func UnPred)
{
    std::string result, tmp;
    while (std::getline(is, tmp) && UnPred(tmp))
        result += tmp + "\n";
    result.pop_back();
    return result;
}

bool is_exist(const std::string& aFileName)
{
    std::ifstream ifs { aFileName };
    return ifs.is_open();
}

bool is_creatable(const std::filesystem::path& aFileName)
{
    std::ofstream ofs { aFileName };
    if (ofs.is_open())
    {
        ofs.close();
        std::filesystem::remove(aFileName);
        return true;
    }
    return false;
}

std::vector<std::string> getTokens(int argc, char** argv)
{
    std::vector<std::string> tokens;
    for (size_t i = 1; i != argc; ++i)
    {
        if (argv[i][0] == '\"')
        {
            std::string tmp { argv[i] + 1 };
            while (tmp.back() != '\"')
            {
                if (++i == argc)
                    throw std::invalid_argument { "missing closing \"" };
                tmp += " ";
                tmp += argv[i];
            }
            tmp.pop_back();
            tokens.emplace_back(std::move(tmp));
        }
        else
            tokens.emplace_back(argv[i]);
    }
    return tokens;
}

std::pair<std::filesystem::path, std::filesystem::path> enterFileName(const std::filesystem::path& aDefaultOut)
{
    std::string rfile, wfile;
    std::cout << "Enter name of file, where will the data read\n>";
    std::getline(std::cin, rfile);
    while (!is_exist(rfile))
    {
        std::cout << "File with this name does not exist or not readable. Try again\n>";
        std::getline(std::cin, rfile);
    }
    std::cout << "Enter the name of the file where the data will be written, or nothing\n>";
    std::getline(std::cin, wfile);
    while (!wfile.empty() && !is_creatable(wfile))
    {
        std::cout << "The file with the same name cannot be created. Try again\n>";
        std::getline(std::cin, wfile);
    }
    if (wfile.empty())
        return { rfile, aDefaultOut };
    return { rfile, wfile };
}

std::pair<std::filesystem::path, std::filesystem::path> getFileName(int argc, char** argv, const std::filesystem::path& aDefaultOut)
{
    if (argc == 1) return enterFileName(aDefaultOut);
    using namespace std::string_literals;
    const auto tokens = getTokens(argc, argv);
    switch (tokens.size())
    {
    case 1:
        if (!is_exist(tokens[0]))
            throw std::invalid_argument { "file with this name ["s + tokens[0] + "] does not exist or not readable" };
        return { tokens[0], aDefaultOut };
    case 2:
        if (!is_exist(tokens[0]))
            throw std::invalid_argument { "file with this name ["s + tokens[0] + "] does not exist or not readable" };
        if (!is_creatable(tokens[1]))
            throw std::invalid_argument { "file with this name ["s + tokens[1] + "] cannot be created" };
        return { tokens[0], tokens[1] };
    default:
        throw std::invalid_argument { "too many arguments (more than two)" };
    }
}

template <typename ClockT>
std::string time_to_str(std::chrono::time_point<ClockT> aPoint, char aDelim = ' ')
{
    using years = std::chrono::duration< int64_t, std::ratio<31556952>>;
    using months = std::chrono::duration< int64_t, std::ratio<2629746>>;
    using days = std::chrono::duration< int64_t, std::ratio<86400>>;
    const auto BEGINNING_OF_AN_EPOCH = 1970;
    const auto Year = std::chrono::duration_cast<years>(aPoint.time_since_epoch()).count() + BEGINNING_OF_AN_EPOCH;
    const auto Month = std::chrono::duration_cast<months>(aPoint.time_since_epoch()).count() % 12 + 1;
    const auto Day = std::chrono::duration_cast<days>(aPoint.time_since_epoch() - std::chrono::duration_cast<months>(aPoint.time_since_epoch())).count() + 1;
    const auto Hour = std::chrono::duration_cast<std::chrono::hours>(aPoint.time_since_epoch()).count() % 24;
    const auto Minute = std::chrono::duration_cast<std::chrono::minutes>(aPoint.time_since_epoch()).count() % 60;
    const auto Second = std::chrono::duration_cast<std::chrono::seconds>(aPoint.time_since_epoch()).count() % 60;
    return (std::ostringstream {} << Day << '.' << Month << '.' << Year % 1000 << aDelim << Hour << 'h' << Minute << 'm' << Second << 's').str();
}

std::filesystem::path generateFileName(std::string_view aPref, std::string_view aSuf)
{
    return (std::ostringstream {} << aPref << time_to_str(std::chrono::system_clock::now(), '_') << aSuf).str();
}

int main(int argc, char** argv)
{
    do
    {
        std::pair<std::filesystem::path, std::filesystem::path> rwfile;
        try
        {
            rwfile = getFileName(argc, argv, generateFileName("Replace_", ".txt"));
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
        auto no_empty = [](std::string_view val) noexcept{return !val.empty(); };
        std::fstream fs { rwfile.first, std::ios::in };
        std::string source_str = read_lines(fs, no_empty);
        std::string old_substr = read_lines(fs, no_empty);
        std::string new_substr = read_lines(fs, no_empty);
        fs.close();
        replace_all(source_str, old_substr, new_substr);
        fs.open(rwfile.second, std::ios::out);
        fs << source_str;
        std::cout << "Replacing occurred successfully\n";
    } while (argc == 1 && keep_on());
}