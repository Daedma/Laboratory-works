#include "..\headers\strategy.hpp"
#include "..\headers\field.hpp"
#include <algorithm>
#include <random>
#include <iterator>

void gameStrategy::strategy_type::shuffle(std::initializer_list<steps_type::value_type> coords, steps_type& dest)
{
    std::vector<steps_type::value_type> temp;
    std::move(coords.begin(), coords.end(), std::back_inserter(temp));
    std::shuffle(temp.begin(), temp.end(), std::random_device {});
    for (auto& i : temp)
    {
        dest.emplace(std::move(i));
    }
}

std::vector<gameStrategy::steps_type::value_type> gameStrategy::strategy_type::enemy_pos(const gameField& arena) const
{
    std::vector<steps_type::value_type> result;
    fieldObjects enemy = Side == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (uint16_t i = 0; i != 3; ++i)
        for (uint16_t j = 0; j != 3; ++j)
            if (arena.at(i, j) == enemy)
                result.emplace_back(i, j);
    return result;
}

gameStrategy::strategy_type::strategy_type(fieldObjects myside) :
    Side { myside } {}

gameStrategy::row::row(uint16_t n, fieldObjects myside) :
    strategy_type { myside }, nRow { n } {}

gameStrategy::row::row(fieldObjects myside) :
    strategy_type { myside }
{
    std::uniform_int_distribution<uint16_t> d { 0, 2 };
    nRow = d(std::random_device {});
}

void gameStrategy::row::fill(gameStrategy::steps_type& dest) const
{
    shuffle({ { nRow, 0 }, { nRow, 1 }, { nRow, 2 } }, dest);
}

bool gameStrategy::row::fit(const gameField& field) const
{
    std::vector<steps_type::value_type> steps { { nRow, 0 }, { nRow, 1 }, { nRow, 2 } };
    auto enemy = enemy_pos(field);
    return std::find_first_of(enemy.cbegin(), enemy.cend(), steps.cbegin(), steps.cend()) == enemy.cend();
}

std::pair<uint16_t, std::unique_ptr<gameStrategy::strategy_type>> gameStrategy::row::suit(const gameField& field, fieldObjects myside)
{
    uint16_t max_count = 0, max_row = 0;
    fieldObjects enemy = myside == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (uint16_t i = 0; i != 3; ++i)
    {
        uint16_t enemy_count = 0, my_count = 0;
        for (uint16_t j = 0; j != 3; ++j)
        {
            if (field.at(i, j) == enemy)
            {
                ++enemy_count;
                break;
            }
            else if (field.at(i, j) == myside)
                ++my_count;
        }
        if (!enemy_count)
        {
            if (my_count > max_count)
            {
                max_row = i;
                max_count = my_count;
            }
        }
    }
    if (max_count)
    {
        return { max_count, std::make_unique<row>(max_row, myside) };
    }
    else
        return { 0, nullptr };
}

gameStrategy::column::column(uint16_t n, fieldObjects myside) :
    strategy_type { myside }, nColumn { n } {}

gameStrategy::column::column(fieldObjects myside) :
    strategy_type { myside }
{
    std::uniform_int_distribution<uint16_t> d { 0, 2 };
    nColumn = d(std::random_device {});
}

void gameStrategy::column::fill(gameStrategy::steps_type& dest) const
{
    shuffle({ { 0, nColumn }, { 1, nColumn }, { 2, nColumn } }, dest);
}

bool gameStrategy::column::fit(const gameField& field) const
{
    std::vector<steps_type::value_type> steps { { 0, nColumn }, { 1, nColumn }, { 2, nColumn } };
    auto enemy = enemy_pos(field);
    return std::find_first_of(enemy.cbegin(), enemy.cend(), steps.cbegin(), steps.cend()) == enemy.cend();
}

std::pair<uint16_t, std::unique_ptr<gameStrategy::strategy_type>> gameStrategy::column::suit(const gameField& field, fieldObjects myside)
{
    uint16_t max_count = 0, max_column = 0;
    fieldObjects enemy = myside == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (uint16_t i = 0; i != 3; ++i)
    {
        uint16_t enemy_count = 0, my_count = 0;
        for (uint16_t j = 0; j != 3; ++j)
        {
            if (field.at(j, i) == enemy)
            {
                ++enemy_count;
                break;
            }
            else if (field.at(j, i) == myside)
                ++my_count;
        }
        if (!enemy_count)
        {
            if (my_count > max_count)
            {
                max_column = i;
                max_count = my_count;
            }
        }
    }
    if (max_count)
    {
        return { max_count, std::make_unique<column>(max_column, myside) };
    }
    else
        return { 0, nullptr };
}

gameStrategy::diagonal::diagonal(gameStrategy::diagonal::types type, fieldObjects) :
    Type { type }
{
    if (Type == types::MAIN)
        Steps = { { 0, 0 }, { 1, 1 }, { 2, 2 } };
    else
        Steps = 
}