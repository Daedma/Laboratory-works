#include "..\headers\strategy.hpp"
#include "..\headers\field.hpp"
#include <algorithm>
#include <vector>
#include <random>
#include <iterator>
#include <set>

const std::array<std::array<gameStrategy::steps_type::value_type, 3>, 8> gameStrategy::Intents =
{
    //rows
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 0U }, gameStrategy::steps_type::value_type { 0U, 1U }, gameStrategy::steps_type::value_type { 0U, 2U } },
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 1U, 0U }, gameStrategy::steps_type::value_type { 1U, 1U }, gameStrategy::steps_type::value_type { 1U, 2U } },
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 2U, 0U }, gameStrategy::steps_type::value_type { 2U, 1U }, gameStrategy::steps_type::value_type { 2U, 2U } },
    //columns
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 0U }, gameStrategy::steps_type::value_type { 1U, 0U }, gameStrategy::steps_type::value_type { 2U, 0U } },
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 1U }, gameStrategy::steps_type::value_type { 1U, 1U }, gameStrategy::steps_type::value_type { 2U, 1U } },
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 2U }, gameStrategy::steps_type::value_type { 1U, 2U }, gameStrategy::steps_type::value_type { 2U, 2U } },
    //diagonals
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 0U }, gameStrategy::steps_type::value_type { 1U, 1U }, gameStrategy::steps_type::value_type { 2U, 2U } },
    std::array<gameStrategy::steps_type::value_type, 3>{ gameStrategy::steps_type::value_type { 0U, 2U }, gameStrategy::steps_type::value_type { 1U, 1U }, gameStrategy::steps_type::value_type { 2U, 0U } }
};

bool gameStrategy::random() const noexcept
{
    return CurStrat == randval;
}

void gameStrategy::clear() noexcept
{
    while (!StepsQueue.empty()) StepsQueue.pop();
}

void gameStrategy::randfill()
{
    std::vector<steps_type::value_type> FreePlaces;
    clear();
    for (uint16_t i = 0U; i != 3U; ++i)
        for (uint16_t j = 0U; j != 3U; ++j)
            if (Arena.at(i, j) == fieldObjects::EMPTY)
                FreePlaces.emplace_back(i, j);
    std::shuffle(FreePlaces.begin(), FreePlaces.end(), std::random_device {});
    for (auto& i : FreePlaces)
        StepsQueue.emplace(i);
}

void gameStrategy::fill()
{
    if (random()) return randfill();
    static std::array<uint16_t, 3> buff { 0, 1, 2 };
    clear();
    std::shuffle(buff.begin(), buff.end(), std::random_device {});
    for (auto i : buff)
        StepsQueue.emplace(Intents[CurStrat][i]);
}

bool gameStrategy::fit() const noexcept
{
    if (random()) return true;
    fieldObjects EnemySide = Side == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (auto& i : Intents[CurStrat])
        if (Arena.at(i.first, i.second) == EnemySide)
            return false;
    return true;
}

std::pair<bool, uint16_t> gameStrategy::priority(size_t nStrat) const noexcept
{
    uint16_t Count = 0U;
    fieldObjects EnemySide = Side == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (auto& i : Intents[nStrat])
    {
        if (Arena.at(i.first, i.second) == EnemySide)
            return { false, -1 };
        else if (Arena.at(i.first, i.second) == Side)
            ++Count;
    }
    return { true, Count };
}

void gameStrategy::change()
{
    std::multiset < std::pair<uint16_t, size_t>, std::greater<std::pair<uint16_t, size_t>>> Priorities;
    std::pair<bool, uint16_t> curval;
    for (size_t i = 0; i != Intents.size(); ++i)
    {
        curval = priority(i);
        if (curval.first)
            Priorities.emplace(curval.second, i);
    }
    if (!Priorities.empty())
        CurStrat = Priorities.begin()->second;
    else
        CurStrat = randval;
}

size_t gameStrategy::select()
{
    static std::default_random_engine e { std::random_device {}() };
    std::vector<size_t> pool;
    for (size_t i = 0; i != Intents.size(); ++i)
    {
        if (priority(i).first)
            pool.emplace_back(i);
    }
    if (pool.empty()) return randval;
    std::uniform_int_distribution<size_t> d { 0, pool.size() - 1 };
    return pool[d(e)];
}

gameStrategy::steps_type::value_type gameStrategy::step() noexcept
{
    steps_type::value_type NextStep;
    do
    {
        NextStep = StepsQueue.top();
        StepsQueue.pop();
    } while (Arena.at(NextStep.first, NextStep.second) != fieldObjects::EMPTY);
    return NextStep;
}

uint16_t gameStrategy::left() const noexcept
{
    if (random()) return static_cast<uint16_t>(-1);
    uint16_t Count = 0U;
    for (const auto& i : Intents[CurStrat])
        if (Arena.at(i.first, i.second) == Side)
            ++Count;
    return 3U - Count;
}

void gameStrategy::reset(fieldObjects _MySide)
{
    Side = _MySide;
    CurStrat = select();
    fill();
}

void gameStrategy::update()
{
    if (!random())
    {
        size_t OldStrat = CurStrat;
        change();
        if (OldStrat != CurStrat)
            fill();
    }
}

gameStrategy::gameStrategy(const gameField& _Field, fieldObjects _MySide) :
    Arena { _Field }, Side { _MySide }, CurStrat { select() }
{
    fill();
}