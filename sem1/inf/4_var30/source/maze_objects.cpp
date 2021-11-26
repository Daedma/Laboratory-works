#include "..\headers\maze_objects.hpp"
#include "..\headers\player.hpp"

MazeObject::MazeObject(int32_t aHp, int32_t aDmg) noexcept :
    hp { aHp }, dmg { aDmg } {}

Pass::Pass() noexcept :
    MazeObject { 0, 0 } {}

std::string_view Pass::take_dmg(Player& aPlayer) noexcept
{
    return "You violently attacked the air in front of you.";
}

std::string_view Pass::try_move(Player& aPlayer, directions aDir) noexcept
{
    aPlayer.move(aDir);
    return "You advanced on.";
}


Wall::Wall(int32_t aHp, int32_t aDmg) noexcept :
    MazeObject { aHp, aDmg } {}

std::string_view Wall::take_dmg(Player& aPlayer) noexcept
{
    aPlayer.break_sword(dmg);
    hp = std::max(hp - aPlayer.deal_dmg(), 0);
    if (hp)
        return "You hit the wall. Your sword's condition decreased.";
    else
        return "You destroyed the wall. Pass is free.";
}

std::string_view Wall::try_move(Player& aPlayer, directions) noexcept
{
    aPlayer.take_dmg(1);
    return "You launched your nose about the wall";
}


Monster::Monster(int32_t aHp, int32_t aDmg) noexcept :
    MazeObject { aHp, aDmg } {}

std::string_view Monster::take_dmg(Player& aPlayer) noexcept
{
    hp = std::max(hp - aPlayer.deal_dmg(), 0);
    if (hp)
        return "You hit the labyrinth monster with your sword.";
    else
        return "You killed a labyrinth monster. Pass is free.";
}

std::string_view Monster::try_move(Player& aPlayer, directions) noexcept
{
    aPlayer.take_dmg(dmg);
    return "You disturbed the labyrinth air, for which he hit you.";
}