#include "headers/maze_game.hpp"
#include "headers/player.hpp"
#include "headers/iotools.hpp"

int main()
{
    do
    {
        try
        {
            auto maze = init_maze();
            auto start_pos = maze.rand_pass();
            if (!start_pos.first)
            {
                std::cout << "There are no passages in the labyrinth. You lose.\n";
                continue;
            }
            std::cout << "The initial coordinates of the player: "
                << start_pos.second.first << ' ' << start_pos.second.second << '\n';
            Player player { start_pos.second, 114, 8 };
            play_maze(player, maze);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error: " << e.what() << '\n';
        }
    } while (keep_on());
}