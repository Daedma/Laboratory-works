#include "..\headers\maze_field.hpp"
#include "..\headers\maze_objects.hpp"
#include "..\headers\Labyrinth.hpp"
#include <fstream>
#include <stdexcept>
#include <iomanip>

template<typename T>
struct default_value;

template<>
struct default_value<Wall>
{
    static constexpr std::pair<int32_t, int32_t> value = { 14, 18 };
};

template<>
struct default_value<Monster>
{
    static constexpr std::pair<int32_t, int32_t> value = { 5, 26 };
};

MazeField::~MazeField() = default;
MazeField::MazeField(MazeField&&) = default;
MazeField& MazeField::operator=(MazeField&&) = default;

MazeField::MazeField(const std::filesystem::path & aPath)
{
    std::ifstream ifs;
    ifs.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
    ifs.open(aPath);
    int64_t nRow, nCol;
    ifs >> nRow >> nCol;
    if (nRow <= 0 || nCol <= 0)
        throw std::invalid_argument { "The number of rows and columns can not be a negative number or null." };
    ifs >> std::ws;
    ifs >> std::noskipws;
    for (size_t i = 0; i != nRow; ++i)
    {
        std::vector<std::unique_ptr<MazeObject>> row;
        for (size_t j = 0; j != nCol; ++j)
        {
            char ch;
            ifs >> ch;
            row.emplace_back(create_obj(ch));
        }
        if (i != nRow - 1 && ifs.get() != '\n')
            throw std::invalid_argument { "Incorrect format." };
        field.emplace_back(std::move(row));
    }
}

MazeField::MazeField(size_t aWidth, size_t aHeight, const std::filesystem::path & aOut)
{
    static constexpr auto default_ex = Labyrinth::exits::hor;
    static constexpr auto default_fill_level = 0.1;
    Labyrinth lab { default_ex, aWidth, aHeight };
    static std::default_random_engine e { std::random_device {}() };
    static std::bernoulli_distribution d { default_fill_level };
    const auto [lnRow, lnCol] = lab.size();
    for (size_t i = 0; i != lnRow; ++i)
    {
        std::vector<std::unique_ptr<MazeObject>> row;
        for (size_t j = 0; j != lnCol; ++j)
        {
            if (lab.at(i, j) == Labyrinth::objects::wall)
                row.emplace_back(
                    std::make_unique<Wall>(default_value<Wall>::value.first, default_value<Wall>::value.second));
            else
            {
                if (d(e))
                    row.emplace_back(rand_obj());
                else
                    row.emplace_back(std::make_unique<Pass>());
            }
        }
    }
}