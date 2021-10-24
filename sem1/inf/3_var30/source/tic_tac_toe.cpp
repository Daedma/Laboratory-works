#include "..\headers\tic_tac_toe.hpp"
#include "..\headers\field.hpp"
#include "..\headers\iotools.hpp"
#include "..\headers\bot.hpp"
#if (defined (_WIN32) || defined (_WIN64))
#define CLEAR_CONSOLE system("cls")
#endif
#if (defined (LINUX) || defined (__linux__))
#define CLEAR_CONSOLE system("clear")
#endif

void print(const gameField& _Arena)
{
    CLEAR_CONSOLE;
    std::cout << "**************** Tic tac toe ******************" << std::endl;
    std::cout << _Arena;
}

bool exodus(gameField& arena)
{
    if (!arena.valid())
    {
        std::cout << "Oops! Draw!" << std::endl;
        arena.reset();
        return true;
    }
    auto vict = arena.check();
    if (vict == fieldObjects::EMPTY)
        return false;
    if (vict == fieldObjects::CROSS)
    {
        std::cout << "Congratulations! CROSS win!" << std::endl;
        arena.reset();
        return true;
    }
    if (vict == fieldObjects::NOUGHT)
    {
        std::cout << "Congratulations! NOUGHT win!" << std::endl;
        arena.reset();
        return true;
    }
    return false;
}

void GameWithBot()
{
    fieldObjects BotSide, PlayerSide;
    std::cout << "Choose your side (1 - crosses, 2 - noughts)\n>";
    if (getstr([](const std::string& val){
        return val == "1" || val == "2";
        }) == "1")
    {
        PlayerSide = fieldObjects::CROSS;
        BotSide = fieldObjects::NOUGHT;
    }
    else
    {
        BotSide = fieldObjects::CROSS;
        PlayerSide = fieldObjects::NOUGHT;
    }
    bool ex = false;
    gameField arena;
    gameBot bot { arena, BotSide };
    auto PlayerTurn = [&arena, PlayerSide](){
        std::cout << "Your turn, enter row and column number\n>";
        auto PlayerCoords = getCoords([&arena](const std::pair<uint16_t, uint16_t>& val){
            return val.first && val.second && val.first < 4 && val.second < 4 && arena.at(val.first - 1, val.second - 1) == fieldObjects::EMPTY;
            });
        if (PlayerSide == fieldObjects::CROSS)
            arena.setX(PlayerCoords.first - 1, PlayerCoords.second - 1);
        else
            arena.setO(PlayerCoords.first - 1, PlayerCoords.second - 1);
    };
    print(arena);
    while (!ex)
    {
        if (PlayerSide == fieldObjects::NOUGHT)
        {
            PlayerTurn();
            print(arena);
            if ((ex = exodus(arena))) return;
            bot.step();
            print(arena);
            ex = exodus(arena);
        }
        else
        {
            bot.step();
            print(arena);
            if ((ex = exodus(arena))) return;
            PlayerTurn();
            print(arena);
            ex = exodus(arena);
        }
    }
}
void GameWith2Players()
{
    gameField arena;
    auto validX = [&arena](const std::pair<uint16_t, uint16_t>& val) noexcept{
        return arena.setX(val.first - 1, val.second - 1);
    };
    auto validO = [&arena](const std::pair<uint16_t, uint16_t>& val) noexcept{
        return arena.setO(val.first - 1, val.second - 1);
    };
    bool ex = false;
    while (!ex)
    {
        print(arena);
        std::cout << "Move NOUGHT. Enter row and column number\n>";
        getCoords(validO);
        print(arena);
        if ((ex = exodus(arena))) break;
        std::cout << "Move CROSS.  Enter row and column number\n>";
        getCoords(validX);
        print(arena);
        ex = exodus(arena);
    }
}

int run()
{
    std::cout << "**************** Tic tac toe ******************\n\n"
        << "Welcome to tic-tac-toe!\n";
    do
    {
        std::cout << "Please, select mode (1 - vs bot, 2 - 2 players)\n>";
        const auto choice = getstr([](const std::string& val){return val == "1" || val == "2"; });
        if (choice == "1")
            GameWithBot();
        else
            GameWith2Players();
    } while (keep_on());
    return 0;
}