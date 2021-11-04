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

void replace_all(std::string& aSource, std::string_view aOld, std::string_view aNew)
{
    size_t cur = aSource.find(aOld);
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

std::pair<std::string, std::string> enterFileName(const std::string& aDefaultOut = std::string { "OUTPUT.TXT" })
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
    if (wfile.empty())
        return { rfile, aDefaultOut };
    return { rfile, wfile };
}

std::pair<std::string, std::string> getFileName(int argc, char** argv, const std::string& aDefaultOut = std::string { "OUTPUT.TXT" })
{
    using namespace std::string_literals;
    switch (argc)
    {
    case 1:
        break;
    case 2:
        if (!is_exist(argv[1]))
            throw std::invalid_argument { "file with this name ["s + argv[1] + "] does not exist or not readable" };
        return { argv[1], aDefaultOut };
    case 3:
        if (!is_exist(argv[1]))
            throw std::invalid_argument { "file with this name ["s + argv[1] + "] does not exist or not readable" };
        return { argv[1], argv[2] };
    default:
        throw std::invalid_argument { "too many arguments (more than two)" };
    }
    return enterFileName(aDefaultOut);
}

int main(int argc, char** argv)
{
    std::pair<std::string, std::string> rwfile;
    try
    {
        rwfile = getFileName(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    auto no_empty = [](const std::string& val) noexcept{return !val.empty(); };
    std::fstream fs { rwfile.first, std::ios::in };
    std::string source_str = read_lines(fs, no_empty);
    std::string old_substr = read_lines(fs, no_empty);
    std::string new_substr = read_lines(fs, no_empty);
    fs.close();
    replace_all(source_str, old_substr, new_substr);
    fs.open(rwfile.second, std::ios::out);
    fs << source_str;
    std::cout << "Replacing occurred successfully\n";
}