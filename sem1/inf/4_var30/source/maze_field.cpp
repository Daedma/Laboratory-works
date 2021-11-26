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
    static constexpr int32_t hp = 14;
    static constexpr int32_t damage = 18;
};

template<>
struct default_value<Monster>
{
    static constexpr int32_t hp = 5;
    static constexpr int32_t damage = 26;
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
    try
    {
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
    catch (const std::exception&)
    {
        field.clear();
        throw;
    }
}


MazeField::MazeField(size_t aWidth, size_t aHeight, const std::filesystem::path & aOut)
{
    static constexpr auto default_ex = Labyrinth::exits::hor;
    static constexpr auto default_fill_level = 0.1;
    static std::default_random_engine e { std::random_device {}() };
    static std::bernoulli_distribution d { default_fill_level };
    Labyrinth lab { default_ex, aWidth, aHeight };
    const auto [lnRow, lnCol] = lab.size();
    try
    {
        for (size_t i = 0; i != lnRow; ++i)
        {
            std::vector<std::unique_ptr<MazeObject>> row;
            for (size_t j = 0; j != lnCol; ++j)
            {
                if (lab.at(i, j) == Labyrinth::objects::wall)
                    row.emplace_back(
                        std::make_unique<Wall>(default_value<Wall>::hp, default_value<Wall>::damage));
                else
                {
                    if (d(e))
                        row.emplace_back(rand_obj());
                    else
                        row.emplace_back(std::make_unique<Pass>());
                }
            }
            field.emplace_back(std::move(row));
        }
    }
    catch (const std::exception&)
    {
        field.clear();
        throw;
    }
    std::ofstream ofs;
    ofs.exceptions(std::ios::failbit | std::ios::badbit);
    ofs.open(aOut);
    ofs << lnRow << ' ' << lnCol << '\n';
    for (const auto& i : field)
    {
        for (const auto& j : i)
            ofs << j->sym();
        ofs << '\n';
    }
}

std::unique_ptr<MazeObject> MazeField::create_obj(char aObjSym) const
{
    using namespace std::string_literals;
    switch (aObjSym)
    {
    case Pass::symbol:
        return std::make_unique<Pass>();
    case Wall::symbol:
        return std::make_unique<Wall>(default_value<Wall>::hp, default_value<Wall>::damage);
    case Monster::symbol:
        return std::make_unique<Monster>(default_value<Monster>::hp, default_value<Monster>::damage);
    default:
        throw std::invalid_argument { "Missing classes that are associated with a symbol \'"s + aObjSym + "\'." };
    }
    return nullptr;
}

std::unique_ptr<MazeObject> MazeField::rand_obj() const
{
    static std::default_random_engine e { std::random_device {}() };
    static std::uniform_int_distribution d { 1, 1 };
    switch (d(e))
    {
    case 1:
        return std::make_unique<Monster>(default_value<Monster>::hp, default_value<Monster>::damage);
    }
    return nullptr;
}

std::pair<bool, std::pair<size_t, size_t>> MazeField::rand_pass() const noexcept
{
    std::vector<std::pair<size_t, size_t>> pool;
    const auto [nRow, nCol] = size();
    for (size_t i = 0; i != nRow; ++i)
        for (size_t j = 0; j != nCol; ++j)
            if (get(nRow, nCol)->sym() == Pass::symbol)
                pool.emplace_back(i, j);
    if (pool.empty())
        return { false, { -1, -1 } };
    std::uniform_int_distribution<size_t> d { 0, pool.size() - 1 };
    return { true, pool[d(std::random_device {})] };
}