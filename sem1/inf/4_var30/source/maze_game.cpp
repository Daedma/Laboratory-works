#include "../headers/maze_game.hpp"
#include "../headers/maze_field.hpp"
#include "../headers/maze_objects.hpp"
#include "../headers/player.hpp"
#include "../headers/iotools.hpp"

namespace
{
    bool won(const Player &aPlayer, const MazeField &aMaze)
    {
        decltype(auto) coord = aPlayer.get_pos();
        decltype(auto) size = aMaze.size();
        return !coord.first || !coord.second || coord.first == size.first - 1 || coord.second == size.second - 1;
    }

    inline Player::coord_t next_pos(Player aPlayer, directions aDir) noexcept
    {
        return aPlayer.move(aDir);
    }
}

void play_in_maze(Player &aPlayer, MazeField &aMaze)
{
    while (true)
    {
        std::cout << "Enter command and direction\n>";
        const auto [command, dir] = getAction([](const auto &val) noexcept
                                              { return (val.first == "attack" || val.first == "move") && val.second >= 1 && val.second <= 4; });
        const auto actionPos = next_pos(aPlayer, static_cast<directions>(dir));
        if (command == "attack")
        {
            aMaze.get(actionPos.first, actionPos.second)->take_dmg(aPlayer);
            if (aMaze.get(actionPos.first, actionPos.second)->need_clear())
                aMaze.clear(actionPos.first, actionPos.second);
        }
        else
        {
            aMaze.get(actionPos.first, actionPos.second)->try_move(aPlayer, static_cast<directions>(dir));
        }
        aPlayer.print_status();
        if (!aPlayer.get_hp())
        {
            /*Вывод сообщения о поражении*/
            return;
        }
        else if (won(aPlayer, aMaze))
        {
            /*Вывод сообщения о победе*/
            return;
        }
    }
}