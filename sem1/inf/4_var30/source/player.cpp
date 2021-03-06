#include "../headers/player.hpp"
#include <algorithm>
#include <iostream>

Player::Player(const coord_t& aCoord, int32_t aHp, int32_t aDmg) noexcept : coord { aCoord }, hp { aHp }, sword { 100, aDmg } {}

const Player::coord_t& Player::move(directions aDir) noexcept
{
    switch (aDir)
    {
    case directions::left:
        --coord.second;
        break;
    case directions::right:
        ++coord.second;
        break;
    case directions::up:
        --coord.first;
        break;
    case directions::down:
        ++coord.first;
        break;
    default:
        break;
    }
    return coord;
}

int32_t Player::break_sword(int32_t aDmg) noexcept
{
    if (sword.cond)
    {
        changes = true;
        if (sword.cond - aDmg <= 0 || !sword.dmg)
        {
            sword.cond = sword.dmg = 0;
            std::cout << "Your sword is broken. Now you will not be able to damage.\n";
        }
        else
        {
            if (aDmg > 0)
                std::cout << "The strength of the sword decreases ...\n";
            else if (aDmg < 0)
                std::cout << "The strength of the sword increases ...\n";
            sword.cond -= aDmg;
            sword.dmg *= sword.cond / 100.;
        }
    }
    return sword.cond;
}

int32_t Player::take_dmg(int32_t aDmg) noexcept
{
    changes = true;
    if (hp)
        hp = std::max(hp - aDmg, 0);
    return hp;
}

int32_t Player::deal_dmg() const noexcept
{
    return sword.dmg;
}

void Player::print_status() const
{
    changes = false;
    std::cout << "HP: " << hp << '\n'
        << "DAMAGE: " << sword.dmg << '\n'
        << "SWORD CONDITION: " << sword.cond << "%\n";
}