#include "..\headers\bot.hpp"
#include "..\headers\field.hpp"

gameBot::gameBot(gameField& Field, fieldObjects BotSide) :
    Arena { Field }, Side { BotSide }, curStrat { Arena, Side } {}

std::pair<bool, gameBot::coord_type> gameBot::is_danger() const noexcept
{
    //все возможные тактики противника
    static constexpr std::array<std::array<coord_type, 3>, 16> enemyTactics =
    {
        //rows
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 0U, 1U }, coord_type { 0U, 2U } },
        std::array<coord_type, 3>{ coord_type { 1U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } },
        std::array<coord_type, 3>{ coord_type { 2U, 0U }, coord_type { 2U, 1U }, coord_type { 2U, 2U } },
        //columns
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 0U }, coord_type { 2U, 0U } },
        std::array<coord_type, 3>{ coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 2U }, coord_type { 2U, 2U } },
        //diagonals
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 2U, 2U } },
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 1U }, coord_type { 2U, 0U } },
        //other
        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 2U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 0U } },
        std::array<coord_type, 3>{ coord_type { 0U, 1U }, coord_type { 1U, 1U }, coord_type { 2U, 2U } },

        std::array<coord_type, 3>{ coord_type { 0U, 2U }, coord_type { 1U, 0U }, coord_type { 1U, 1U } },
        std::array<coord_type, 3>{ coord_type { 2U, 2U }, coord_type { 1U, 0U }, coord_type { 1U, 1U } },
        std::array<coord_type, 3>{ coord_type { 0U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } },
        std::array<coord_type, 3>{ coord_type { 2U, 0U }, coord_type { 1U, 1U }, coord_type { 1U, 2U } }
    };
    fieldObjects EnemySide = Side == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
    for (const auto& i : enemyTactics)
    {
        uint16_t Count = 0;
        coord_type LastClear;
        for (const auto& j : i)
        {
            if (Arena.at(j.first, j.second) == EnemySide)
                ++Count;
            else if (Arena.at(j.first, j.second) == Side)
            {
                Count = 0;
                break;
            }
            else
                LastClear = j;
        }
        if (Count == 2)
            return { true, LastClear };
    }
    return { false, { -1, -1 } };
}

void gameBot::step()
{
    auto setObject = Side == fieldObjects::CROSS ? &gameField::setX : &gameField::setO;
    auto is_dungerous = is_danger();
    curStrat.update();
    if (is_dungerous.first && curStrat.left() != 1)//если есть опасность и осталось больше 1 хода до победы бота
        (Arena.*setObject)(is_dungerous.second.first, is_dungerous.second.second);//помешать противнику выиграть
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