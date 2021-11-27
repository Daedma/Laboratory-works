#include "../headers/maze_game.hpp"
#include "../headers/maze_objects.hpp"
#include "../headers/player.hpp"
#include "../headers/iotools.hpp"
#include <fstream>

namespace{
    bool won(const Player& aPlayer, const MazeField& aMaze)
    {
        decltype(auto) coord = aPlayer.get_pos();
        decltype(auto) size = aMaze.size();
        return !coord.first || !coord.second || coord.first == size.first - 1 || coord.second == size.second - 1;
    }

    inline Player::coord_t next_pos(Player aPlayer, directions aDir) noexcept
    {
        return aPlayer.move(aDir);
    }


    //проверка на существование файла
    bool is_exist(const std::string& aFileName)
    {
        std::ifstream ifs { aFileName };
        return ifs.is_open();
    }


    //проверка на возможность создания файла с таким именем
    bool is_creatable(const std::filesystem::path& aFileName)
    {
        std::ofstream ofs { aFileName };
        if (ofs.is_open())
        {
            ofs.close();
            std::filesystem::remove(aFileName);
            return true;
        }
        return false;
    }

    std::filesystem::path getInFileName()
    {
        std::cout << "Enter the name of the file from where you read\n>";
        std::string s;
        while (!std::getline(std::cin, s) || !is_exist(s))
            std::cout << "The file with the same name does not exist. Try again\n>";
        return s;
    }

    std::filesystem::path getOutFileName()
    {
        std::cout << "Enter the name of the file from where you write\n>";
        std::string s;
        while (!std::getline(std::cin, s) || !is_creatable(s))
            std::cout << "The file with the same name cannot be created. Try again\n>";
        return s;
    }

    void print(const Player& aPlayer, const MazeField& aField)
    {
        const auto szMaze = aField.size();
        decltype(auto) plPos = aPlayer.get_pos();
        for (size_t i = 0; i != szMaze.first; ++i)
        {
            for (size_t j = 0; j != szMaze.second; ++j)
            {
                if (plPos.first == i && plPos.second == j)
                    std::cout << 'h';
                else
                    std::cout << aField.get(i, j)->sym();
            }
            std::cout << '\n';
        }
    }
}

MazeField init_maze()
{
    std::cout << "You want to read(1) an existing labyrinth from a file or generate(2) a new?\n>";
    int choice = geti([](char val) noexcept{return val == 1 || val == 2; });
    if (choice == 1)
        return MazeField(getInFileName());
    else
    {
        std::cout << "Enter number of row\n>";
        auto nRow = geti([](int val) noexcept{return val > 0; });
        std::cout << "Enter number of column\n>";
        auto nColumn = geti([](int val) noexcept{return val > 0; });
        return MazeField(nColumn, nRow, getOutFileName());
    }
}

void play_maze(Player& aPlayer, MazeField& aMaze)
{
    std::cout << "Directions:\n1 - right\n2 - up\n3 - left\n4 - down\n";
    aPlayer.print_status();
    while (true)
    {
        //print(aPlayer, aMaze);
        std::cout << "Enter command and direction\n>";
        const auto [command, dir] = getAction([](const auto& val) noexcept{ return (val.first == "attack" || val.first == "move") && val.second >= 1 && val.second <= 4; });
        const auto actionPos = next_pos(aPlayer, static_cast<directions>(dir - 1));
        if (command == "attack")
        {
            std::cout << aMaze.get(actionPos.first, actionPos.second)->take_dmg(aPlayer) << '\n';
            std::cout << aMaze.get(actionPos.first, actionPos.second)->info_hp();
            if (aMaze.get(actionPos.first, actionPos.second)->need_clear())
                aMaze.clear(actionPos.first, actionPos.second);
        }
        else
        {
            std::cout << aMaze.get(actionPos.first, actionPos.second)->try_move(aPlayer, static_cast<directions>(dir - 1)) << '\n';
        }
        if (aPlayer.changed())
            aPlayer.print_status();
        if (!aPlayer.get_hp())
        {
            std::cout << "You lost and died in a labyrinth.\n";
            return;
        }
        else if (won(aPlayer, aMaze))
        {
            std::cout << "You got out of the labyrinth.\n";
            return;
        }
    }
}