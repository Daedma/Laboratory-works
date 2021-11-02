#include "../headers/tic_tac_toe.hpp"
#include "../headers/field.hpp"
#include "../headers/iotools.hpp"
#include "../headers/bot.hpp"
#include <algorithm>
#include <random>
#include <functional>
#include <thread>
#include <chrono>
#ifdef __GNUC__
#define IGNORE_RETURN_VALUE auto USELESS_VARIABLE_ =
#endif
#ifndef __GNUC__
#define IGNORE_RETURN_VALUE (void)
#endif
#if (defined(_WIN32) || defined(_WIN64))
#define CLEAR_CONSOLE \
    IGNORE_RETURN_VALUE system("cls")
#endif
#if (defined(LINUX) || defined(__linux__))
#define CLEAR_CONSOLE \
    IGNORE_RETURN_VALUE system("clear")
#endif

namespace{
    void print(const gameField& _Arena) noexcept
    {
        CLEAR_CONSOLE;
        std::cout << "**************** Tic tac toe ******************" << std::endl;
        std::cout << _Arena;
    }

    //проверить поле на наличие победителя
    bool exodus(gameField& arena)
    {
        auto vict = arena.check();
        if (vict == fieldObjects::CROSS)
        {
            std::cout << "Congratulations! CROSS win!" << std::endl;
            return true;
        }
        if (vict == fieldObjects::NOUGHT)
        {
            std::cout << "Congratulations! NOUGHT win!" << std::endl;
            return true;
        }
        if (!arena.valid())
        {
            std::cout << "Oops! Draw!" << std::endl;
            return true;
        }
        return false;
    }

    //вывести текст с задержкой между выводами отдельных сиволов строки
    template <typename Rep, typename Period>
    void slow_print(const std::string& aMessage, const std::chrono::duration<Rep, Period>& aInterval) noexcept
    {
        for (auto i : aMessage)
        {
            std::this_thread::sleep_for(aInterval);
            std::cout << i;
            std::cout.flush();
        }
    }

    //создать функцию для проверки поля на наличие победителя
    auto create_ex_checker(const gameBot& aBot, const gameField& aArena) noexcept
    {
        return [&aBot, &aArena]() noexcept{
            using namespace std::chrono_literals;
            auto winner = aArena.check();
            if (aBot.get_side() == winner)
            {
                std::cout << "The bot won you!\nThe bot says: ";
                slow_print("Ha ha ha! I'm smarter than you! I have 16 megabytes of memory!\n", 50ms);
                return true;
            }
            else if (winner != fieldObjects::EMPTY)
            {
                std::cout << "Congratulations, you won! You are smarter than a computer!\nThe bot says: ";
                slow_print("No! You defeated me! I'm sure it’s foul play or your luck involved.", 50ms);
                return true;
            }
            else if (!aArena.valid())
            {
                std::cout << "Draw!\n";
                return true;
            }
            return false;
        };
    }

    //случайное распределение сторон
    fieldObjects rand_distribute() noexcept
    {
        static std::default_random_engine e { std::random_device {}() };
        static std::uniform_int_distribution<> d { 0, 1 };
        return static_cast<fieldObjects>(d(e));
    }

    //распределить стороны
    void distribute(fieldObjects& botSide, fieldObjects& playerSide) noexcept
    {
        static const std::array<std::string, 23> answers = {
            "1", "Crosses", "crosses", "X", "x", "cross", "Cross", "CROSS", "CROSSES",
            "Rand", "rand", "random", "Random",
            "2", "Nought", "nought", "O", "o", "0", "nought", "Nought", "NOUGHT", "NOUGHT" };
        static const auto endCross = std::find(answers.cbegin(), answers.cend(), "Rand");
        static const auto endRand = std::find(answers.cbegin(), answers.cend(), "2");

        std::cout << "Choose your side (1 - crosses, 2 - noughts)\n>";
        decltype(answers)::const_iterator choice;
        getstr([&choice](const std::string& val) noexcept{
            choice = std::find(answers.cbegin(), answers.cend(), val);
            return choice != answers.cend();
            });
        if (choice < endCross)
        {
            playerSide = fieldObjects::CROSS;
            botSide = fieldObjects::NOUGHT;
        }
        else if (choice < endRand)
        {
            playerSide = rand_distribute();
            botSide = playerSide == fieldObjects::CROSS ? fieldObjects::NOUGHT : fieldObjects::CROSS;
        }
        else
        {
            playerSide = fieldObjects::NOUGHT;
            botSide = fieldObjects::CROSS;
        }
    }

    //создать функцию для проведения хода определенной стороны
    template <fieldObjects ObjT>
    std::enable_if_t<ObjT != fieldObjects::EMPTY, std::function<void(void)>> get_move(gameBot& aBot, gameField& aArena) noexcept
    {
        if (aBot.get_side() == ObjT)
        {
            return [&aBot](){ aBot.step(); };
        }
        else
        {
            return [&aArena](){
                std::cout << "Your turn, enter row and column number\n>";
                auto PlayerCoords = getCoords([&aArena](const std::pair<uint16_t, uint16_t>& val)                { return val.first && val.second && val.first < 4 && val.second < 4 && aArena.at(val.first - 1, val.second - 1) == fieldObjects::EMPTY; });
                aArena.setObj(ObjT, PlayerCoords.first - 1, PlayerCoords.second - 1);
            };
        }
    }

    void GameWithBot()
    {
        using namespace std::chrono_literals;
        fieldObjects BotSide, PlayerSide;
        distribute(BotSide, PlayerSide);
        gameField arena;
        gameBot bot { arena, BotSide };
        auto nought_move = get_move<fieldObjects::NOUGHT>(bot, arena);
        auto cross_move = get_move<fieldObjects::CROSS>(bot, arena);
        auto has_winner = create_ex_checker(bot, arena);
        std::cout << "The bot says: ";
        slow_print("I will beat you meat brains! Ha ha!\n", 40ms);
        print(arena);
        while (true)
        {
            nought_move();
            print(arena);
            if (has_winner())
                return;
            cross_move();
            print(arena);
            if (has_winner())
                return;
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
        print(arena);
        while (true)
        {
            std::cout << "Move NOUGHT. Enter row and column number\n>";
            getCoords(validO);
            print(arena);
            if (exodus(arena))
                return;
            std::cout << "Move CROSS.  Enter row and column number\n>";
            getCoords(validX);
            print(arena);
            if (exodus(arena))
                return;
        }
    }
}
int run()
{
    std::cout << "**************** Tic tac toe ******************\n\n"
        << "Welcome to tic-tac-toe!\n";
    do
    {
        std::cout << "Please, select mode (1 - vs bot, 2 - 2 players)\n>";
        const auto choice = getstr([](const std::string& val){ return val == "1" || val == "2"; });
        if (choice == "1")
            GameWithBot();
        else
            GameWith2Players();
    } while (keep_on());
    return 0;
}