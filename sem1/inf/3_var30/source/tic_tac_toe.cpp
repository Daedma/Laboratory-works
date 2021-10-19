#include "..\headers\tic_tac_toe.hpp"
#include "..\headers\field.hpp"
#include "..\headers\iotools.hpp"
#if (defined (_WIN32) || defined (_WIN64))
#define CLEAR_CONSOLE system("cls")
#endif
#if (defined (LINUX) || defined (__linux__))
#define CLEAR_CONSOLE system("clear")
#endif

void print(const ttField& _Arena)
{
    CLEAR_CONSOLE;
    std::cout << "**************** Tic tac toe ******************" << std::endl;
    std::cout << _Arena;
}

bool exodus(ttField& arena)
{
    if (!arena.valid())
    {
        std::cout << "Oops! Draw!" << std::endl;
        arena.reset();
        return true;
    }
    auto vict = arena.check();
    if (vict == ttObjType::EMPTY)
        return false;
    if (vict == ttObjType::CROSS)
    {
        std::cout << "Congratulations! CROSS win!" << std::endl;
        arena.reset();
        return true;
    }
    if (vict == ttObjType::NOUGHT)
    {
        std::cout << "Congratulations! NOUGHT win!" << std::endl;
        arena.reset();
        return true;
    }
    return false;
}

int run()
{
    ttField arena;
    auto validX = [&arena](const std::pair<uint16_t, uint16_t>& val) noexcept{
        return arena.setX(val.second, val.first);
    };
    auto validO = [&arena](const std::pair<uint16_t, uint16_t>& val) noexcept{
        return arena.setO(val.second, val.first);
    };
    bool ex = false;
    do
    {
        while (!ex)
        {
            print(arena);
            std::cout << "Move NOUGHT. Enter row and column number\n>";
            getXY(validO);
            print(arena);
            if ((ex = exodus(arena))) break;
            std::cout << "Move CROSS.  Enter row and column number\n>";
            getXY(validX);
            print(arena);
            ex = exodus(arena);
        }
    } while (keep_on());
    return 0;
}