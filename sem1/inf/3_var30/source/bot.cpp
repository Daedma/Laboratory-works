#include "..\headers\bot.hpp"
#include "..\headers\field.hpp"
#include <iostream>

gameBot::gameBot(gameField& Field, fieldObjects BotSide) :
    Arena { Field }, Side { BotSide }, curStrat { Arena, Side } {}

std::pair<bool, gameBot::coord_type> gameBot::danger() const noexcept
{
    static const std::array<std::array<coord_type, 3>, 16> enemyTactics =
    {
        //rows
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 0U, 1U }, coord_type { 0U, 2U } },
        std::array<coord_type, 3>{ coord_type { 1U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } },
        std::array<coord_type, 3>{ coord_type { 2U, 0U }, coord_type { 2U, 1U }, coord_type { 2U, 2U } },
        //columns
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 0U }, coord_type { 2U, 0U } },
        std::array<coord_type, 3> { coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 2U }, coord_type { 2U, 2U } },
        //diagonals
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 2U, 2U } },
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 1U }, coord_type { 2U, 0U } },
        //other
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 0U } },
        std::array<coord_type, 3> { coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 2U } },

        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 0U }, coord_type { 1U, 1U } },
        std::array<coord_type, 3> { coord_type { 2U, 2U }, coord_type { 1U, 0U }, coord_type { 1U, 1U } },
        std::array<coord_type, 3> { coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } },
        std::array<coord_type, 3> { coord_type { 2U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } }
    };
    fieldObjects EnemySide = Side == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (const auto& i : enemyTactics)
    {
        uint16_t Count = 0;
        coord_type LastClear = i[2];
        for (const auto& j : i)
        {
            if (Arena.at(j.first, j.second) == EnemySide)
            {
                if (++Count == 2 && Arena.at(LastClear.first, LastClear.second) == fieldObjects::EMPTY)
                    return { true, LastClear };
            }
            else if (Arena.at(j.first, j.second) == Side)
                break;
            else
                LastClear = j;
        }
    }
    return { false, { -1, -1 } };
}

void gameBot::step()
{
    auto setObject = Side == fieldObjects::CROSS ? &gameField::setX : &gameField::setO;
    auto is_dungerous = danger();
    curStrat.update();
    if (is_dungerous.first && curStrat.left() != 1)
        (Arena.*setObject)(is_dungerous.second.first, is_dungerous.second.second);
    else
    {
        auto Step = curStrat.step();
        (Arena.*setObject)(Step.first, Step.second);
    }
}

void gameBot::reset(fieldObjects NewSide)
{
    Side = NewSide;
    Arena.reset();
    curStrat.reset(NewSide);
}

void gameBot::reset()
{
    reset(Side);
}