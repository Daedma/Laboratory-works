#include "./working_shedule.hpp"
#include "./iotools.hpp"
#include <fstream>
#include <vector>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <memory>

//проверка на существование файла
bool is_exist(const std::filesystem::path& aFileName)
{
    std::ifstream ifs { aFileName };
    return ifs.is_open();
}

//проверка на возможность создания файла с таким именем
bool is_creatable(const std::filesystem::path& aFileName)
{
    if (is_exist(aFileName))
        return true;
    std::ofstream ofs { aFileName };
    if (ofs.is_open())
    {
        ofs.close();
        std::filesystem::remove(aFileName);
        return true;
    }
    return false;
}

std::filesystem::path get_input_filename()
{
    std::cout << "Enter the name of the file to read\n>";
    std::string rfile;
    std::getline(std::cin, rfile);
    while (!is_exist(rfile))
    {
        std::cout << "File with such a name does not exist or it is not available for reading. Try again\n>";
        std::getline(std::cin, rfile);
    }
    return rfile;
}

std::filesystem::path get_output_filename()
{
    std::cout << "Enter the name of the file to write\n>";
    std::string wfile;
    std::getline(std::cin, wfile);
    while (!is_creatable(wfile))
    {
        std::cout << "The file with the same name cannot be created. Try again\n>";
        std::getline(std::cin, wfile);
    }
    return wfile;
}

template<typename OutputIt>
void read_info(OutputIt aDest)
{
    std::ifstream ifs { get_input_filename() };
    std::copy(std::istream_iterator<Work_schedule>{ifs}, {}, aDest);
    if (ifs.fail() && !ifs.eof())
        throw std::exception { "data does not match the format" };
}

template<typename OutputIt>
void enter_info(OutputIt aDest)
{
    std::cout << "Enter the number of items you are going to enter\n>";
    int n = geti([](int val) noexcept{return val > 0; });
    for (size_t i = 0; i != n; ++i)
    {
        std::cout << "Enter the " << i + 1 << "st element\n>";
        *aDest++ = getValue<Work_schedule>();
    }
}

template<typename OutputIt>
void get_info(OutputIt aDest)
{
    std::cout << "Do you want to read the data from the file(1) from entering them manually(2)?\n>";
    int choice = geti([](int val) noexcept{return val == 1 || val == 2; });
    if (choice == 1)
        read_info(aDest);
    else
        enter_info(aDest);
}

std::unique_ptr<std::ostream> get_out()
{
    std::cout << "Do you want to display the results in a file(1) or in the console(2)?\n>";
    int choice = geti([](int val) noexcept{return val == 1 || val == 2; });
    if (choice == 1)
        return std::make_unique<std::ofstream>(get_output_filename());
    else
        return std::make_unique<std::ostream>(std::cout.rdbuf());
}

int main()
{
    do
    {
        try
        {
            std::vector<Work_schedule> work_info;
            get_info(std::back_inserter(work_info));
            std::vector<Time> time_info;
            std::transform(work_info.cbegin(), work_info.cend(), std::back_inserter(time_info),
                [](const Work_schedule& val) noexcept{return val.working_hours(); });
            auto out = get_out();
            for (auto [entered, calculated] = std::make_pair(work_info.cbegin(), time_info.cbegin());
                entered != work_info.cend(); ++entered, ++calculated)
            {
                *out << "Date: " << entered->get_date() << '\n'
                    << "Arrival: " << entered->get_arrival() << '\n'
                    << "Leaving: " << entered->get_leaving() << '\n'
                    << "Working time: " << *calculated << "\n\n";
                //*out << *entered << "\n\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error: " << e.what() << '\n';
        }
    } while (keep_on());

}